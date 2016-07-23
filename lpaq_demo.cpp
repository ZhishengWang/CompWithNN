/*
  Source code for one paper in sips 2016.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include<iostream>
//#define NDEBUG  // remove for debugging
#include <assert.h>
//_CRT_SECURE_NO_WARNINGS
#define PROGNAME "lpaq_demo"
//#define _CRT_SECURE_NO_WARNINGS

using namespace std;

// Create an array p of n elements of type T
template <class T> void alloc(T*&p, int n) {
	p = (T*)calloc(n, sizeof(T));
	if (!p) printf("Out of memory\n"), exit(1);
}

// 8, 16, 32 bit unsigned types (adjust as appropriate)
typedef unsigned char  U8;
typedef unsigned short U16;
typedef unsigned int   U32;

// Read/write a 4 byte big-endian number
int get4(FILE* in) {
	int r = getc(in);
	r = (r << 8) + getc(in);
	r = (r << 8) + getc(in);
	r = (r << 8) + getc(in);
	return r;
}

void put4(U32 c, FILE* out) {
	fprintf(out, "%c%c%c%c", c >> 24, c >> 16, c >> 8, c);
}

///////////////////////////// Squash //////////////////////////////

// return p = 1/(1 + exp(-d)), d scaled by 8 bits, p scaled by 12 bits
int squash(int d) {
	static const int t[33] = {
		1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101,
		1546, 2047, 2549, 2994, 3348, 3607, 3785, 3901, 3975, 4022,
		4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094 };
	if (d>2047) return 4095;
	if (d<-2047) return 0;
	int w = d & 127;
	d = (d >> 7) + 16;
	return (t[d] * (128 - w) + t[(d + 1)] * w + 64) >> 7;
}

//////////////////////////// Stretch ///////////////////////////////

// Inverse of squash. stretch(d) returns ln(p/(1-p)), d scaled by 8 bits,
// p by 12 bits.  d has range -2047 to 2047 representing -8 to 8.  
// p has range 0 to 4095 representing 0 to 1.

class Stretch {
	short t[4096];
public:
	Stretch();
	int operator()(int p) const {
		assert(p >= 0 && p<4096);
		return t[p];
	}
} stretch;

Stretch::Stretch() {
	int pi = 0;
	for (int x = -2047; x <= 2047; ++x) {  // invert squash()
		int i = squash(x);
		for (int j = pi; j <= i; ++j)
			t[j] = x;
		pi = i + 1;
	}
	t[4095] = 2047;
}

///////////////////////// state table ////////////////////////

// State table:
//   nex(state, 0) = next state if bit y is 0, 0 <= state < 256
//   nex(state, 1) = next state if bit y is 1
//
// States represent a bit history within some context.
// State 0 is the starting state (no bits seen).
// States 1-30 represent all possible sequences of 1-4 bits.
// States 31-252 represent a pair of counts, (n0,n1), the number
//   of 0 and 1 bits respectively.  If n0+n1 < 16 then there are
//   two states for each pair, depending on if a 0 or 1 was the last
//   bit seen.
// If n0 and n1 are too large, then there is no state to represent this
// pair, so another state with about the same ratio of n0/n1 is substituted.
// Also, when a bit is observed and the count of the opposite bit is large,
// then part of this count is discarded to favor newer data over old.

static const U8 State_table[256][2] = {
	{ 1, 2 }, { 3, 5 }, { 4, 6 }, { 7, 10 }, { 8, 12 }, { 9, 13 }, { 11, 14 }, // 0
	{ 15, 19 }, { 16, 23 }, { 17, 24 }, { 18, 25 }, { 20, 27 }, { 21, 28 }, { 22, 29 }, // 7
	{ 26, 30 }, { 31, 33 }, { 32, 35 }, { 32, 35 }, { 32, 35 }, { 32, 35 }, { 34, 37 }, // 14
	{ 34, 37 }, { 34, 37 }, { 34, 37 }, { 34, 37 }, { 34, 37 }, { 36, 39 }, { 36, 39 }, // 21
	{ 36, 39 }, { 36, 39 }, { 38, 40 }, { 41, 43 }, { 42, 45 }, { 42, 45 }, { 44, 47 }, // 28
	{ 44, 47 }, { 46, 49 }, { 46, 49 }, { 48, 51 }, { 48, 51 }, { 50, 52 }, { 53, 43 }, // 35
	{ 54, 57 }, { 54, 57 }, { 56, 59 }, { 56, 59 }, { 58, 61 }, { 58, 61 }, { 60, 63 }, // 42
	{ 60, 63 }, { 62, 65 }, { 62, 65 }, { 50, 66 }, { 67, 55 }, { 68, 57 }, { 68, 57 }, // 49
	{ 70, 73 }, { 70, 73 }, { 72, 75 }, { 72, 75 }, { 74, 77 }, { 74, 77 }, { 76, 79 }, // 56
	{ 76, 79 }, { 62, 81 }, { 62, 81 }, { 64, 82 }, { 83, 69 }, { 84, 71 }, { 84, 71 }, // 63
	{ 86, 73 }, { 86, 73 }, { 44, 59 }, { 44, 59 }, { 58, 61 }, { 58, 61 }, { 60, 49 }, // 70
	{ 60, 49 }, { 76, 89 }, { 76, 89 }, { 78, 91 }, { 78, 91 }, { 80, 92 }, { 93, 69 }, // 77
	{ 94, 87 }, { 94, 87 }, { 96, 45 }, { 96, 45 }, { 48, 99 }, { 48, 99 }, { 88, 101 }, // 84
	{ 88, 101 }, { 80, 102 }, { 103, 69 }, { 104, 87 }, { 104, 87 }, { 106, 57 }, { 106, 57 }, // 91
	{ 62, 109 }, { 62, 109 }, { 88, 111 }, { 88, 111 }, { 80, 112 }, { 113, 85 }, { 114, 87 }, // 98
	{ 114, 87 }, { 116, 57 }, { 116, 57 }, { 62, 119 }, { 62, 119 }, { 88, 121 }, { 88, 121 }, // 105
	{ 90, 122 }, { 123, 85 }, { 124, 97 }, { 124, 97 }, { 126, 57 }, { 126, 57 }, { 62, 129 }, // 112
	{ 62, 129 }, { 98, 131 }, { 98, 131 }, { 90, 132 }, { 133, 85 }, { 134, 97 }, { 134, 97 }, // 119
	{ 136, 57 }, { 136, 57 }, { 62, 139 }, { 62, 139 }, { 98, 141 }, { 98, 141 }, { 90, 142 }, // 126
	{ 143, 95 }, { 144, 97 }, { 144, 97 }, { 68, 57 }, { 68, 57 }, { 62, 81 }, { 62, 81 }, // 133
	{ 98, 147 }, { 98, 147 }, { 100, 148 }, { 149, 95 }, { 150, 107 }, { 150, 107 }, { 108, 151 }, // 140
	{ 108, 151 }, { 100, 152 }, { 153, 95 }, { 154, 107 }, { 108, 155 }, { 100, 156 }, { 157, 95 }, // 147
	{ 158, 107 }, { 108, 159 }, { 100, 160 }, { 161, 105 }, { 162, 107 }, { 108, 163 }, { 110, 164 }, // 154
	{ 165, 105 }, { 166, 117 }, { 118, 167 }, { 110, 168 }, { 169, 105 }, { 170, 117 }, { 118, 171 }, // 161
	{ 110, 172 }, { 173, 105 }, { 174, 117 }, { 118, 175 }, { 110, 176 }, { 177, 105 }, { 178, 117 }, // 168
	{ 118, 179 }, { 110, 180 }, { 181, 115 }, { 182, 117 }, { 118, 183 }, { 120, 184 }, { 185, 115 }, // 175
	{ 186, 127 }, { 128, 187 }, { 120, 188 }, { 189, 115 }, { 190, 127 }, { 128, 191 }, { 120, 192 }, // 182
	{ 193, 115 }, { 194, 127 }, { 128, 195 }, { 120, 196 }, { 197, 115 }, { 198, 127 }, { 128, 199 }, // 189
	{ 120, 200 }, { 201, 115 }, { 202, 127 }, { 128, 203 }, { 120, 204 }, { 205, 115 }, { 206, 127 }, // 196
	{ 128, 207 }, { 120, 208 }, { 209, 125 }, { 210, 127 }, { 128, 211 }, { 130, 212 }, { 213, 125 }, // 203
	{ 214, 137 }, { 138, 215 }, { 130, 216 }, { 217, 125 }, { 218, 137 }, { 138, 219 }, { 130, 220 }, // 210
	{ 221, 125 }, { 222, 137 }, { 138, 223 }, { 130, 224 }, { 225, 125 }, { 226, 137 }, { 138, 227 }, // 217
	{ 130, 228 }, { 229, 125 }, { 230, 137 }, { 138, 231 }, { 130, 232 }, { 233, 125 }, { 234, 137 }, // 224
	{ 138, 235 }, { 130, 236 }, { 237, 125 }, { 238, 137 }, { 138, 239 }, { 130, 240 }, { 241, 125 }, // 231
	{ 242, 137 }, { 138, 243 }, { 130, 244 }, { 245, 135 }, { 246, 137 }, { 138, 247 }, { 140, 248 }, // 238
	{ 249, 135 }, { 250, 69 }, { 80, 251 }, { 140, 252 }, { 249, 135 }, { 250, 69 }, { 80, 251 }, // 245
	{ 140, 252 }, { 0, 0 }, { 0, 0 }, { 0, 0 } };  // 252
#define nex(state,sel) State_table[state][sel]

//////////////////////////// StateMap, APM //////////////////////////

// A StateMap maps a context to a probability.  Methods:
//
// Statemap sm(n) creates a StateMap with n contexts using 4*n bytes memory.
// sm.p(y, cx, limit) converts state cx (0..n-1) to a probability (0..4095).
//     that the next y=1, updating the previous prediction with y (0..1).
//     limit (1..1023, default 1023) is the maximum count for computing a
//     prediction.  Larger values are better for stationary sources.

class StateMap {
protected:
	const int N;  // Number of contexts
	int cxt;      // Context of last prediction
	U32 *t;       // cxt -> prediction in high 22 bits, count in low 10 bits
	static int dt[1024];  // i -> 16K/(i+3)
	void update(int y, int limit) {
		assert(cxt >= 0 && cxt<N);
		int n = t[cxt] & 1023, p = t[cxt] >> 10;  // count, prediction
		if (n<limit) ++t[cxt];
		else t[cxt] = t[cxt] & 0xfffffc00 | limit;
		t[cxt] += (((y << 22) - p) >> 3)*dt[n] & 0xfffffc00;
	}
public:
	StateMap(int n = 256);

	// update bit y (0..1), predict next bit in context cx
	int p(int y, int cx, int limit = 1023) {
		assert(y >> 1 == 0);
		assert(cx >= 0 && cx<N);
		assert(limit>0 && limit<1024);
		update(y, limit);
		return t[cxt = cx] >> 20;
	}
};

int StateMap::dt[1024] = { 0 };

StateMap::StateMap(int n) : N(n), cxt(0) {
	alloc(t, N);
	for (int i = 0; i<N; ++i)
		t[i] = 1 << 31;
	if (dt[0] == 0)
		for (int i = 0; i<1024; ++i)
			dt[i] = 16384 / (i + i + 3);
}

// An APM maps a probability and a context to a new probability.  Methods:
//
// APM a(n) creates with n contexts using 96*n bytes memory.
// a.pp(y, pr, cx, limit) updates and returns a new probability (0..4095)
//     like with StateMap.  pr (0..4095) is considered part of the context.
//     The output is computed by interpolating pr into 24 ranges nonlinearly
//     with smaller ranges near the ends.  The initial output is pr.
//     y=(0..1) is the last bit.  cx=(0..n-1) is the other context.
//     limit=(0..1023) defaults to 255.

class APM : public StateMap {
public:
	APM(int n);
	int pp(int y, int pr, int cx, int limit = 255) {
		assert(y >> 1 == 0);
		assert(pr >= 0 && pr<4096);
		assert(cx >= 0 && cx<N / 24);
		assert(limit>0 && limit<1024);
		int cx_ = cx;
		update(y, limit);
		pr = (stretch(pr) + 2048) * 23;
		int wt = pr & 0xfff;  // interpolation weight of next element
		cx = cx * 24 + (pr >> 12);
		assert(cx >= 0 && cx<N - 1);
		pr = (t[cx] >> 13)*(0x1000 - wt) + (t[cx + 1] >> 13)*wt >> 19;
		cxt = cx + (wt >> 11);
		//if (cx == cx_ | (cx_ == cx + 1)) printf("666\n");
		return pr;
	}
};

APM::APM(int n) : StateMap(n * 24) {
	for (int i = 0; i<N; ++i) {
		int p = ((i % 24 * 2 + 1) * 4096) / 48 - 2048;
		t[i] = (U32(squash(p)) << 20) + 6;
	}
}

//////////////////////////// Mixer /////////////////////////////

// Mixer m(N, M) combines models using M neural networks with
//     N inputs each using 4*M*N bytes of memory.  It is used as follows:
// m.update(y) trains the network where the expected output is the
//     last bit, y.
// m.add(stretch(p)) inputs prediction from one of N models.  The
//     prediction should be positive to predict a 1 bit, negative for 0,
//     nominally -2K to 2K.
// m.set(cxt) selects cxt (0..M-1) as one of M neural networks to use.
// m.p() returns the output prediction that the next bit is 1 as a
//     12 bit number (0 to 4095).  The normal sequence per prediction is:
//
// - m.add(x) called N times with input x=(-2047..2047)
// - m.set(cxt) called once with cxt=(0..M-1)
// - m.p() called once to predict the next bit, returns 0..4095
// - m.update(y) called once for actual bit y=(0..1).

inline void train(int *t, int *w, int n, int err) {
	for (int i = 0; i < n; ++i){
		w[i] += t[i] * err >> 16;
		if (w[i] >= 65535) w[i] = 65535;
		if (w[i] <= -65536) w[i] = -65536;  //constrained to 16 bit 
	}

}

inline int dot_product(int *t, int *w, int n) {
	int sum = 0;
	for (int i = 0; i<n; ++i)
		sum += t[i] * w[i];
	return sum >> 8;
}

class Mixer {
	const int N, M;  // max inputs, max contexts
	int* tx;         // N inputs
	int wx[7 * 80];         // N*M weights
	int cxt;         // context
	int nx;          // Number of inputs in tx, 0 to N
	int pr;          // last result (scaled 12 bits)
public:
	Mixer(int n, int m);

	// Adjust weights to minimize coding cost of last prediction
	void update(int y) {
		int err = ((y << 12) - pr) * 7;
		assert(err >= -32768 && err<32768);
		train(&tx[0], &wx[cxt*N], N, err);
		nx = 0;
	}

	// Input x (call up to N times)
	void add(int x) {
		assert(nx<N);
		tx[nx++] = x;
	}

	// Set a context
	void set(int cx) {
		assert(cx >= 0 && cx<M);
		cxt = cx;
	}

	// predict next bit
	int p() {
		return pr = squash(dot_product(&tx[0], &wx[cxt*N], N) >> 8);
	}
};

Mixer::Mixer(int n, int m) :
N(n), M(m), tx(0)/*, wx(0)*/, cxt(0), nx(0), pr(2048) {
	assert(n>0 && N>0 && M>0);
	alloc(tx, N);
	*wx = { 0 };
	//alloc(wx, N*M);
}

//////////////////////////// HashTable /////////////////////////

// A HashTable maps a 32-bit index to an array of B bytes.
// The first byte is a checksum using the upper 8 bits of the
// index.  The second byte is a priority (0 = empty) for hash
// replacement.  The index need not be a hash.

// HashTable<B> h(n) - create using n bytes  n and B must be 
//     powers of 2 with n >= B*4, and B >= 2.
// h[i] returns array [1..B-1] of bytes indexed by i, creating and
//     replacing another element if needed.  Element 0 is the
//     checksum and should not be modified.
unsigned long hash_count[9] = { 0 };  //test for hash-function

template <int B>
class HashTable {
	
public:
	U8* t;  // table: 1 element = B bytes: checksum priority data data
	const int N;  // size in bytes
	HashTable(int n);
	U8* operator[](U32 i);
};

template <int B>
HashTable<B>::HashTable(int n) : t(0), N(n) {
	assert(B >= 2 && (B&B - 1) == 0);
	assert(N >= B * 4 && (N&N - 1) == 0);
	alloc(t, N + B * 4 + 64);
	t += 64 - int(((long)t) & 63);  // align on cache line boundary
	memset(t, 0, N + B * 4); //all initialized to zero
}

template <int B>
inline U8* HashTable<B>::operator[](U32 i) {
	i *= 123456791;
	i = i << 16 | i >> 16;
	i *= 234567891;
	int chk = i >> 24;
	i = (i * 16)&(N - 16);
	if (t[i] == chk) { hash_count[0]++; return t + i; }
	if (t[i + 16] == chk) { hash_count[1]++;  return t + i + 16; }
	if (t[i + 32] == chk) { hash_count[2]++; return t + i + 32; }

	//if check sum failed：
	if (t[i + 1] > t[i + 16 + 1] || t[i + 1] > t[i + 32 + 1]){
		i += 16; hash_count[3]++;
	}
	if (t[i + 1] > t[i + 16 + 1]) { i += 16; hash_count[4]++; }
	//memset(t + i, 0, 16);
	t[i] = chk;
	return t + i;
}

//////////////////////////// MatchModel ////////////////////////

// MatchModel(n) predicts next bit using most recent context match.
//     using n bytes of memory.  n must be a power of 2 at least 8.
// MatchModel::p(y, m) updates the model with bit y (0..1) and writes
//     a prediction of the next bit to Mixer m.  It returns the length of
//     context matched (0..62).

class MatchModel {
	const int N;  // last buffer index, n/2-1
	const int HN; // last hash table index, n/8-1
	enum { MAXLEN = 20 };   // maximum match length, at most 62
	U8* buf;    // input buffer
	int* ht;    // context hash -> next byte in buf
	int pos;    // number of bytes in buf
	int match;  // pointer to current byte in matched context in buf
	int len;    // length of match
	U32 h1, h2; // context hashes
	int c0;     // last 0-7 bits of y
	int bcount; // number of bits in c0 (0..7)
	int cxt;
	StateMap sm;  // len, bit, last byte -> prediction
public:
	MatchModel(int n);  // n must be a power of 2 at least 8.
	int p(int y, Mixer& m);  // update bit y (0..1), predict next bit to m
};

MatchModel::MatchModel(int n) : N(n / 2 - 1), HN(n / 8 - 1), buf(0), ht(0), pos(0),
match(0), len(0), h1(0), h2(0), c0(1), bcount(0), cxt(0), sm(34<< 8) {
	assert(n >= 8 && (n&n - 1) == 0);
	alloc(buf, N + 1);
	alloc(ht, HN + 1);
}

int MatchModel::p(int y, Mixer& m) {

	// update context
	c0 += c0 + y;
	++bcount;
	if (bcount == 8) {
		bcount = 0;
		h1 = h1*(3 << 3) + c0&HN;
		//h2 = h2*(5 << 5) + c0&HN;
		buf[pos++] = c0;
		c0 = 1;
		pos &= N;

		// find or extend match
		if (len>0) {
			++match;
			match &= N;
			if (len<MAXLEN) ++len;
		}
		else {
			match = ht[h1];
			if (match != pos) {
				int i;
				while (len<MAXLEN && (i = match - len - 1 & N) != pos
					&& buf[i] == buf[pos - len - 1 & N])
					++len;
			}
		}
		/*if (len<2) {
			len = 0;
			match = ht[h2];
			if (match != pos) {
				int i;
				while (len<MAXLEN && (i = match - len - 1 & N) != pos
					&& buf[i] == buf[pos - len - 1 & N])
					++len;
			}
		}*/

		// update index
		ht[h1] = pos;
		//ht[h2] = pos;
	}

	// predict
	//int cxt = c0 + (1<<(bcount&7)); //a little better!a little.
	//int cxt = c0;
	if (len>0 && (buf[match] + 256 >> 8 - bcount) == c0) {
		int b = buf[match] >> 7 - bcount & 1;  // next bit
		//if (len<16) cxt = len * 2 + b;
		//else cxt = (len >> 2) * 2 + b + 24;
		cxt = len * 2 + b;
		cxt = cxt * 256 + buf[pos - 1 & N];
	}
	else{
		len = 0;
		cxt = c0;
	}

	m.add(stretch(sm.p(y, cxt)));

	return len;
}

//////////////////////////// Predictor /////////////////////////

// A Predictor estimates the probability that the next bit of
// uncompressed data is 1.  Methods:
// Predictor(n) creates with 3*n bytes of memory.
// p() returns P(1) as a 12 bit number (0-4095).
// update(y) trains the predictor with the actual bit (0 or 1).

int MEM = 0;  // Global memory usage = 3*MEM bytes (1<<20 .. 1<<29)

class Predictor {
	int pr;  // next prediction
	int len;
	int order;
public:
	Predictor();
	int p() const { assert(pr >= 0 && pr<4096); return pr; }
	void update(int y);
};

Predictor::Predictor() : pr(2048), len(0), order(0) {}

void Predictor::update(int y) {
	static U8 cxt0[0x10000];  // order 1 cxt -> state
	//static U8 cxt[5][0x1000000];
	static HashTable<16> t1(MEM);  // cxt -> state
	static HashTable<16> t2(MEM);  // cxt -> state
	static HashTable<16> t3(MEM);  // cxt -> state
	static HashTable<16> t4(MEM);  // cxt -> state
	static HashTable<16> t5(MEM);  // cxt -> state
	static int c0 = 1;  // last 0-7 bits with leading 1
	static int c4 = 0;  // last 4 bytes
	static U8 *addr[6] = { cxt0, t1.t, t2.t, t3.t, t4.t, t5.t };  // pointer to bit history
	//static U8 *addr[6] = { cxt0, cxt[1], cxt[2], cxt[3], cxt[4], cxt[5] };
	static int bcount = 0;  // bit count
	static StateMap sm[6];
	static APM a1(0x100)/*, a2(0x4000)*/;
	static U32 hash[6];
	static Mixer m(7, 80);
	static MatchModel mm(MEM);  // predicts next bit by matching context
	//static int test_count[6] = { 0 };
	assert(MEM>0);

	// update model
	assert(y == 0 || y == 1);
	*addr[0] = nex(*addr[0], y);
	*addr[1] = nex(*addr[1], y);
	*addr[2] = nex(*addr[2], y);
	*addr[3] = nex(*addr[3], y);
	*addr[4] = nex(*addr[4], y);
	*addr[5] = nex(*addr[5], y);
	m.update(y);

	// update context
	++bcount;
	c0 += c0 + y;
	if (bcount == 8) {
		c0 -= 256;
		c4 = c4 << 8 | c0;
		hash[0] = c0 << 8;  // order 1
		hash[1] = (c4 & 0xffff) << 5;  // order 2
		hash[2] = (c4 << 8) * 3;  // order 3
		hash[3] = c4 * 5;  // order 4
		hash[4] = hash[4] * (11 << 5) + c0 * 13 &0x3fffffff;  // order 6
		if (c0 >= 65 && c0 <= 90) c0 += 32;  // lowercase unigram word order
		if (c0 >= 97 && c0 <= 122) hash[5] = (hash[5] + c0)*(7 << 3);
		else hash[5] = 0;
		addr[1] = t1[hash[1]] + 1;
		addr[2] = t2[hash[2]] + 1;
		addr[3] = t3[hash[3]] + 1;
		addr[4] = t4[hash[4]] + 1;
		addr[5] = t5[hash[5]] + 1;
		c0 = 1;
		bcount = 0;
	}
	if (bcount == 4) {
		addr[1] = t1[hash[1] + c0] + 1;
		addr[2] = t2[hash[2] + c0] + 1;
		addr[3] = t3[hash[3] + c0] + 1;
		addr[4] = t4[hash[4] + c0] + 1;
		addr[5] = t5[hash[5] + c0] + 1;
	}
	else if (bcount>0) {
		int j = y + 1 << (bcount & 3) - 1;
		addr[1] += j;
		addr[2] += j;
		addr[3] += j;
		addr[4] += j;
		addr[5] += j;
	}
	addr[0] = cxt0 + hash[0] + c0;



	// predict
	len = mm.p(y, m);
	//len = 0;
	order = 0;
	if (len == 0) {
		if (*addr[4]) ++order;
		if (*addr[3]) ++order;
		if (*addr[2]) ++order;
		if (*addr[1]) ++order;
	}
	else order = 5 +  (len >= 4) + (len >= 8) + (len >= 12) + (len>=16);
	int s0 = stretch(sm[0].p(y, *addr[0]));
	int s1 = stretch(sm[1].p(y, *addr[1]));
	int s2 = stretch(sm[2].p(y, *addr[2]));
	int s3 = stretch(sm[3].p(y, *addr[3]));
	int s4 = stretch(sm[4].p(y, *addr[4]));
	int s5 = stretch(sm[5].p(y, *addr[5]));
	m.add(s0);
	m.add(s1);
	m.add(s2);
	m.add(s3);
	m.add(s4);
	m.add(s5);
	//m.set(order + 10 * (hash[0] >> 13));
	m.set(order + 10 * (hash[0] >> 13));
	pr = m.p();
	//m.update(y);
	int pr1 = a1.pp(y, pr, c0);
	pr = pr + 3 * pr1 >> 2;
	//pr = pr + 3 * a2.pp(y, pr, c0^hash[0] >> 2) >> 2;
}

Predictor predictor;
//////////////////////////// Encoder ////////////////////////////

// An Encoder does arithmetic encoding.  Methods:
// Encoder(COMPRESS, f) creates encoder for compression to archive f, which
//     must be open past any header for writing in binary mode.
// Encoder(DECOMPRESS, f) creates encoder for decompression from archive f,
//     which must be open past any header for reading in binary mode.
// code(i) in COMPRESS mode compresses bit i (0 or 1) to file f.
// code() in DECOMPRESS mode returns the next decompressed bit from file f.
// compress(c) in COMPRESS mode compresses one byte.
// decompress() in DECOMPRESS mode decompresses and returns one byte.
// flush() should be called exactly once after compression is done and
//     before closing f.  It does nothing in DECOMPRESS mode.
//int lcount = 0;

typedef enum { COMPRESS, DECOMPRESS } Mode;
class Encoder {
private:
	const Mode mode;       // Compress or decompress?
	FILE* archive;         // Compressed data file
	U32 x1, x2;            // Range, initially [0, 1), scaled by 2^32
	U32 x;                 // Decompress mode: last 4 input bytes of archive
	enum { BUFSIZE = 0x20000 }; // 1 Mb 进行一次 block 更新，减少文件读取速度
	static unsigned char* buf; // Compression output buffer, size BUFSIZE
	int usize, csize;      // Buffered uncompressed and compressed sizes

	// Compress bit y or return decompressed bit
	int code(int y = 0) {
		int p = predictor.p();
		assert(p >= 0 && p<4096);
		p += p<2048;
		U32 xmid = x1 + (x2 - x1 >> 12)*p + (((x2 - x1) & 0xfff)*p >> 12);
		assert(xmid >= x1 && xmid<x2);
		if (mode == DECOMPRESS) y = x <= xmid;
		y ? (x2 = xmid) : (x1 = xmid + 1);
		predictor.update(y);

		/*if (lcount <= 20){
			lcount++;
			printf("low:%x high:%x xmid:%x y:%d pr:%x\n", x1, x2, xmid, y, p);
		}*/
		while (((x1^x2) & 0xff000000) == 0) {  // pass equal leading bytes of range
			if (mode == COMPRESS) {
				buf[csize++] = x2 >> 24; //if (lcount <= 20){ printf("compressed data:%d\n", x2 >> 24); }
			}
			x1 <<= 8;
			x2 = (x2 << 8) + 255;
			if (mode == DECOMPRESS) x = (x << 8) + getc(archive);
		}
		return y;
	}

public:
	double usum, csum;     // Total of usize, csize
	Encoder(Mode m, FILE* f);
	void flush();  // call this when compression is finished

	// Compress one byte
	void compress(int c) {
		assert(mode == COMPRESS);
		++usize;
		for (int i = 7; i >= 0; --i)
			code((c >> i) & 1);
		if (csize>BUFSIZE - 256)
			flush();
	}

	// Decompress and return one byte
	int decompress() {
		++usize;
		int c = 0;
		for (int i = 0; i<8; ++i)
			c += c + code();
		return c;
	}
};
unsigned char* Encoder::buf = 0;

Encoder::Encoder(Mode m, FILE* f) :
mode(m), archive(f), x1(0), x2(0xffffffff), x(0),
usize(0), csize(0), usum(0), csum(0) {
	if (mode == DECOMPRESS) {  // x = first 4 bytes of archive
		for (int i = 0; i<4; ++i)
			x = (x << 8) + (getc(archive) & 255);
		csize = 4;
	}
	else if (!buf)
		alloc(buf, BUFSIZE);
}

void Encoder::flush() {
	if (mode == COMPRESS) {
		buf[csize++] = x1 >> 24;
		buf[csize++] = 255;
		buf[csize++] = 255;
		buf[csize++] = 255;
		putc(0, archive);
		putc('c', archive);
		put4(usize, archive);
		put4(csize, archive);
		fwrite(buf, 1, csize, archive);
		usum += usize;
		csum += csize + 10;
		printf("%12.0f -> %12.0f"
			"\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b",
			usum, csum);
		x1 = x = usize = csize = 0;
		x2 = 0xffffffff;
	}
}

static unsigned long total_usize = 0;  //总的已压缩文件大小

void lpaq1(FILE* in, FILE* out, Mode mode) {
	if (mode == COMPRESS) {
		Encoder e(COMPRESS, out);
		//Encoder *e = new Encoder(COMPRESS, out);
		int c;
		while ((c = getc(in)) != EOF)
			e.compress(c);
		e.flush();
		total_usize += (long)e.usum;
		//delete e;
	}
	else {
		int usize = get4(in);
		get4(in);  // csize
		//Encoder *e = new Encoder(COMPRESS, out);
		Encoder e(DECOMPRESS, in);
		while (usize--) {
			int c = e.decompress();
			if (out) putc(c, out);
		}
		//delete e;
	}
}


///////////////////////////// Store ///////////////////////////

// Store a file as: {'\0' mode usize csize contents}...
void store(FILE* in, FILE* out) {
	assert(in);
	assert(out);

	// Store in blocks
	const int BLOCKSIZE = 0x100000; // 128KB/block---->encoder 处<<4
	//const int BLOCKSIZE = 1 << 17;
	static char* buf = 0;
	if (!buf) alloc(buf, BLOCKSIZE);
	bool first = true;
	while (true) {
		int n = fread(buf, 1, BLOCKSIZE, in);
		if (!first && n <= 0) break;
		fprintf(out, "%c%c", 0, 's');
		put4(n, out);  // usize
		put4(n, out);  // csize
		fwrite(buf, 1, n, out);
		first = false;
	}

	// Close file
	fclose(in);
}

// Write usize == csize bytes of an uncompressed block from in to out
void unstore(FILE* in, FILE* out) {
	assert(in);
	int usize = get4(in);
	int csize = get4(in);
	if (usize != csize)
		printf("Bad archive format: usize=%d csize=%d\n", usize, csize);
	static char* buf = 0;
	const int BUFSIZE = 0x1000;
	if (!buf) alloc(buf, BUFSIZE);
	while (csize>0) {
		usize = csize;
		if (usize>BUFSIZE) usize = BUFSIZE;
		if (int(fread(buf, 1, usize, in)) != usize)
			printf("Unexpected end of archive\n"), exit(1);
		if (out) fwrite(buf, 1, usize, out);
		csize -= usize;
	}
}

//////////////////////// Archiving functions ////////////////////////

const int MAXNAMELEN = 1023;  // max filename length

// Compress a file to out
void compress(const char* filename, FILE* out, int option) {

	// Open input file
	FILE* in = fopen(filename, "rb");
	if (!in) {
		printf("File not found: %s\n", filename);
		return;
	}
	fprintf(out, "%s", filename);
	printf("%s ", filename);

	// Compress depending on option
	if (option == 's')
		store(in, out);
	else if (option == 'c')
		lpaq1(in, out, COMPRESS);
	printf("\n");
	fclose(in);
}

// List archive contents
void list(FILE *in) {
	double usum = 0, csum = 0;  // uncompressed and compressed size per file
	double utotal = 0, ctotal = 4;  // total size in archive
	static char filename[MAXNAMELEN + 1];
	int mode = 0;

	while (true) {

		// Get filename, mode
		int c = getc(in);
		if (c == EOF) break;
		if (c) {   // start of new file?  Print previous file
			if (mode)
				printf("%10.0f -> %10.0f %c %s\n", usum, csum, mode, filename);
			int len = 0;
			filename[len++] = c;
			while ((c = getc(in)) != EOF && c)
				if (len<MAXNAMELEN) filename[len++] = c;
			filename[len] = 0;
			utotal += usum;
			ctotal += csum;
			usum = 0;
			csum = len;
		}

		// Get uncompressed size
		mode = getc(in);
		int usize = get4(in);
		usum += usize;

		// Get compressed size
		int csize = get4(in);
		csum += csize + 10;

		if (usize<0 || csize<0 || mode != 'c' && mode != 's')
			printf("Archive corrupted usize=%d csize=%d mode=%d at %ld\n",
			usize, csize, mode, ftell(in)), exit(1);

		// Skip csize bytes
		const int BUFSIZE = 0x1000;
		char buf[BUFSIZE];
		while (csize>BUFSIZE)
			csize -= fread(buf, 1, BUFSIZE, in);
		fread(buf, 1, csize, in);
	}
	printf("%10.0f -> %10.0f %c %s\n", usum, csum, mode, filename);
	utotal += usum;
	ctotal += csum;
	printf("%10.0f -> %10.0f total\n", utotal, ctotal);
}

// Return true if the first 4 bytes of in are a valid archive
bool check_archive(FILE* in) {
	return getc(in) == 'l' && getc(in) == 'P' && getc(in) == 'q' && getc(in) == 1;
}

// Open archive and check for valid archive header, exit if bad
FILE* open_archive(const char* filename) {
	FILE* in = fopen(filename, "rb");
	if (!in)
		printf("Cannot find archive %s\n", filename), exit(1);
	if (!check_archive(in)) {
		fclose(in);
		printf("%s: Not a lpq1 archive\n", filename);
		exit(1);
	}
	return in;
}

// Extract files given command line arguments
// Input is: [filename {'\0' mode usize csize contents}...]...
void extract(int argc, char** argv) {
	assert(argc>2);
	assert(argv[1][0] == 'x');
	static char filename[MAXNAMELEN + 1];  // filename from archive

	// Open archive
	FILE* in = open_archive(argv[2]);

	// Extract files
	argc -= 3;
	argv += 3;
	FILE* out = 0;
	while (true) {  // for each block

		// Get filename
		int c;
		for (int i = 0;; ++i) {
			c = getc(in);
			if (c == EOF) break;
			if (i<MAXNAMELEN) filename[i] = c;
			if (!c) break;
		}
		if (c == EOF) break;

		// Open output file
		if (filename[0]) {  // new file?
			const char* fn = filename;
			if (argc>0) fn = argv[0], --argc, ++argv;
			if (out) fclose(out);
			out = fopen(fn, "rb");
			if (out) {
				printf("\nCannot overwrite file, skipping: %s ", fn);
				fclose(out);
				out = 0;
			}
			else {
				out = fopen(fn, "wb");
				if (!out) printf("\nCannot create file: %s ", fn);
			}
			if (out) {
				if (fn == filename) printf("\n%s ", filename);
				else printf("\n%s -> %s ", filename, fn);
			}
		}

		// Extract block
		int mode = getc(in);
		if (mode == 's')
			unstore(in, out);
		else if (mode == 'c')
			lpaq1(in, out, DECOMPRESS);
		else
			printf("\nUnsupported compression mode %c %d at %ld\n",
			mode, mode, ftell(in)), exit(1);
	}
	printf("\n");
	if (out) fclose(out);
}

// Command line is: lpq1 {a|x|l} archive [[-option] files...]...
int main(int argc, char** argv) {
	clock_t start = clock();

	if ((argv[1][0] == '-') && (argv[1][1] <= '9') && (argv[1][1] >= '0')){
		MEM = 1 << (18 + argv[1][1] - '0');
		argc--;
		argv++;
	}
	else
		MEM = 1 << 18;
	// Check command line arguments
	if (argc<3 || argv[1][1] || (argv[1][0] != 'a' && argv[1][0] != 'x'
		&& argv[1][0] != 'l') || (argv[1][0] == 'a' && argc<4) || argv[2][0] == '-')
	{
		printf("lpq1 archiver (C) 2007, Matt Mahoney\n"
			"Free software under GPL, http://www.gnu.org/copyleft/gpl.html\n"
			"\n"
			"To create archive: lpq1 a archive [[-s|-c] files...]...\n"
			"  -s = store, -c = compress (default)\n"
			"To extract files:  lpq1 x archive [files...]\n"
			"To list contents:  lpq1 l archive\n");
		exit(1);
	}

	// Create archive
	if (argv[1][0] == 'a') {
		int option = 'c';  // -c or -s
		//FILE* out = fopen(argv[2], "rb");
		//if (out) printf("Cannot overwrite archive %s\n", argv[2]), exit(1);
		FILE* out = fopen(argv[2], "wb");
		if (!out) printf("Cannot create archive %s\n", argv[2]), exit(1);
		fprintf(out, "lPq%c", 1);
		for (int i = 3; i<argc; ++i) {
			if (argv[i][0] == '-' && (argv[i][1] == 'c' || argv[i][1] == 's')
				&& argv[i][2] == 0)
				option = argv[i][1];
			else {
				compress(argv[i], out, option);
			}
		}
		double time = double(clock() - start) / CLOCKS_PER_SEC;
		printf("%ld -> %ld in %1.2f sec, %1.2f KB/sec,%1.5lf \n", (long)total_usize, ftell(out),
			time, total_usize / (time*1000.0),(double)1.0*ftell(out)*8/total_usize);
		total_usize = 0;
	}

	// List archive contents
	else if (argv[1][0] == 'l') {
		FILE* in = open_archive(argv[2]);
		list(in);
		fclose(in);
	}

	// Extract from archive
	else if (argv[1][0] == 'x') {
		extract(argc, argv);
		printf("%1.2f sec \n", double(clock() - start) / CLOCKS_PER_SEC);
	}

	printf("\n count: \n");
	for (int i = 0; i < 6; ++i){
		printf("%ld ", hash_count[i]);
	}
	return 0;
}
