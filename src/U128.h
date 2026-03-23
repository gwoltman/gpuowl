/*******************************************************************
*
*    Author: Kareem Omar
*    kareem.h.omar@gmail.com
*    https://github.com/komrad36
*
*    Last updated Feb 15, 2021
*******************************************************************/

#pragma once

#include <cstdint>
#include <intrin.h>
#include <iosfwd>

using I8  = int8_t;
using I16 = int16_t;
using I32 = int32_t;
using I64 = int64_t;

using U8  = uint8_t;
using U16 = uint16_t;
using U32 = uint32_t;
using U64 = uint64_t;

#define MAKE_BINARY_OP_HELPERS(op) \
friend auto operator op(const U128& x, U8   y) { return operator op(x, (U128)y); }  \
friend auto operator op(const U128& x, U16  y) { return operator op(x, (U128)y); }  \
friend auto operator op(const U128& x, U32  y) { return operator op(x, (U128)y); }  \
friend auto operator op(const U128& x, U64  y) { return operator op(x, (U128)y); }  \
friend auto operator op(const U128& x, I8   y) { return operator op(x, (U128)y); }  \
friend auto operator op(const U128& x, I16  y) { return operator op(x, (U128)y); }  \
friend auto operator op(const U128& x, I32  y) { return operator op(x, (U128)y); }  \
friend auto operator op(const U128& x, I64  y) { return operator op(x, (U128)y); }  \
friend auto operator op(const U128& x, char y) { return operator op(x, (U128)y); }  \
friend auto operator op(U8   x, const U128& y) { return operator op((U128)x, y); }  \
friend auto operator op(U16  x, const U128& y) { return operator op((U128)x, y); }  \
friend auto operator op(U32  x, const U128& y) { return operator op((U128)x, y); }  \
friend auto operator op(U64  x, const U128& y) { return operator op((U128)x, y); }  \
friend auto operator op(I8   x, const U128& y) { return operator op((U128)x, y); }  \
friend auto operator op(I16  x, const U128& y) { return operator op((U128)x, y); }  \
friend auto operator op(I32  x, const U128& y) { return operator op((U128)x, y); }  \
friend auto operator op(I64  x, const U128& y) { return operator op((U128)x, y); }  \
friend auto operator op(char x, const U128& y) { return operator op((U128)x, y); }

#define MAKE_BINARY_OP_HELPERS_FLOAT(op) \
friend auto operator op(const U128& x, float  y) { return (float)x op y; }   \
friend auto operator op(const U128& x, double y) { return (double)x op y; }  \
friend auto operator op(float  x, const U128& y) { return x op (float)y; }    \
friend auto operator op(double x, const U128& y) { return x op (double)y; }

#define MAKE_BINARY_OP_HELPERS_U64(op) \
friend U128 operator op(const U128& x, U8  n) { return operator op(x, (U64)n); }    \
friend U128 operator op(const U128& x, U16 n) { return operator op(x, (U64)n); }    \
friend U128 operator op(const U128& x, U32 n) { return operator op(x, (U64)n); }    \
friend U128 operator op(const U128& x, I8  n) { return operator op(x, (U64)n); }    \
friend U128 operator op(const U128& x, I16 n) { return operator op(x, (U64)n); }    \
friend U128 operator op(const U128& x, I32 n) { return operator op(x, (U64)n); }    \
friend U128 operator op(const U128& x, I64 n) { return operator op(x, (U64)n); }    \
friend U128 operator op(const U128& x, const U128& n) { return operator op(x, (U64)n); }

class U128
{
public:
    friend U128 DivMod(U128 n, U128 d, U128& rem);

    U128() = default;
    U128(U8    x) : m_lo(x), m_hi(0) {}
    U128(U16   x) : m_lo(x), m_hi(0) {}
    U128(U32   x) : m_lo(x), m_hi(0) {}
    U128(U64   x) : m_lo(x), m_hi(0) {}
    U128(I8    x) : m_lo(I64(x)), m_hi(I64(x) >> 63) {}
    U128(I16   x) : m_lo(I64(x)), m_hi(I64(x) >> 63) {}
    U128(I32   x) : m_lo(I64(x)), m_hi(I64(x) >> 63) {}
    U128(I64   x) : m_lo(I64(x)), m_hi(I64(x) >> 63) {}
    U128(U64 hi, U64 lo) : m_lo(lo), m_hi(hi) {}

    // inexact values truncate, as per the Standard [conv.fpint]
    // passing values unrepresentable in the destination format is undefined behavior,
    // as per the Standard, but this implementation saturates
    U128(float x);

    // inexact values truncate, as per the Standard [conv.fpint]
    // passing values unrepresentable in the destination format is undefined behavior,
    // as per the Standard, but this implementation saturates
    U128(double x);

    U128& operator+=(const U128& x)
    {
        static_cast<void>(_addcarry_u64(_addcarry_u64(0, m_lo, x.m_lo, &m_lo), m_hi, x.m_hi, &m_hi));
        return *this;
    }

    friend U128 operator+(const U128& x, const U128& y)
    {
        U128 ret;
        static_cast<void>(_addcarry_u64(_addcarry_u64(0, x.m_lo, y.m_lo, &ret.m_lo), x.m_hi, y.m_hi, &ret.m_hi));
        return ret;
    }

    MAKE_BINARY_OP_HELPERS(+);
    MAKE_BINARY_OP_HELPERS_FLOAT(+);

    U128& operator-=(const U128& x)
    {
        static_cast<void>(_subborrow_u64(_subborrow_u64(0, m_lo, x.m_lo, &m_lo), m_hi, x.m_hi, &m_hi));
        return *this;
    }

    friend U128 operator-(const U128& x, const U128& y)
    {
        U128 ret;
        static_cast<void>(_subborrow_u64(_subborrow_u64(0, x.m_lo, y.m_lo, &ret.m_lo), x.m_hi, y.m_hi, &ret.m_hi));
        return ret;
    }

    MAKE_BINARY_OP_HELPERS(-);
    MAKE_BINARY_OP_HELPERS_FLOAT(-);

    U128& operator*=(const U128& x)
    {
        // ab * cd
        // ==
        // (2^64*a + b) * (2^64*c + d)
        // if a*c == e, a*d == f, b*c == g, b*d == h
        // |ee|ee|  |  |
        // |  |fg|fg|  |
        // |  |  |hh|hh|

        U64 hHi;
        const U64 hLo = _umul128(m_lo, x.m_lo, &hHi);
        m_hi = hHi + m_hi * x.m_lo + m_lo * x.m_hi;
        m_lo = hLo;
        return *this;
    }

    friend U128 operator*(const U128& x, const U128& y)
    {
        U128 ret;
        U64 hHi;
        ret.m_lo = _umul128(x.m_lo, y.m_lo, &hHi);
        ret.m_hi = hHi + y.m_hi * x.m_lo + y.m_lo * x.m_hi;
        return ret;
    }

    MAKE_BINARY_OP_HELPERS(*);
    MAKE_BINARY_OP_HELPERS_FLOAT(*);

    U128& operator/=(const U128& x)
    {
        U128 rem;
        *this = DivMod(*this, x, rem);
        return *this;
    }

    friend U128 operator/(const U128& x, const U128& y)
    {
        U128 rem;
        return DivMod(x, y, rem);
    }

    MAKE_BINARY_OP_HELPERS(/);
    MAKE_BINARY_OP_HELPERS_FLOAT(/);

    U128& operator%=(const U128& x)
    {
        static_cast<void>(DivMod(*this, x, *this));
        return *this;
    }

    friend U128 operator%(const U128& x, const U128& y)
    {
        U128 ret;
        static_cast<void>(DivMod(x, y, ret));
        return ret;
    }

    MAKE_BINARY_OP_HELPERS(%);

    U128& operator&=(const U128& x)
    {
        m_hi &= x.m_hi;
        m_lo &= x.m_lo;
        return *this;
    }

    friend U128 operator&(const U128& x, const U128& y)
    {
        return U128(x.m_hi & y.m_hi, x.m_lo & y.m_lo);
    }

    MAKE_BINARY_OP_HELPERS(&);

    U128& operator|=(const U128& x)
    {
        m_hi |= x.m_hi;
        m_lo |= x.m_lo;
        return *this;
    }

    friend U128 operator|(const U128& x, const U128& y)
    {
        return U128(x.m_hi | y.m_hi, x.m_lo | y.m_lo);
    }

    MAKE_BINARY_OP_HELPERS(|);

    U128& operator^=(const U128& x)
    {
        m_hi ^= x.m_hi;
        m_lo ^= x.m_lo;
        return *this;
    }

    friend U128 operator^(const U128& x, const U128& y)
    {
        return U128(x.m_hi ^ y.m_hi, x.m_lo ^ y.m_lo);
    }

    MAKE_BINARY_OP_HELPERS(^);

    U128& operator>>=(U64 n)
    {
        const U64 lo = __shiftright128(m_lo, m_hi, (U8)n);
        const U64 hi = m_hi >> (n & 63ULL);

        m_lo = n & 64 ? hi : lo;
        m_hi = n & 64 ? 0  : hi;

        return *this;
    }

    friend U128 operator>>(const U128& x, U64 n)
    {
        U128 ret;

        const U64 lo = __shiftright128(x.m_lo, x.m_hi, (U8)n);
        const U64 hi = x.m_hi >> (n & 63ULL);

        ret.m_lo = n & 64 ? hi : lo;
        ret.m_hi = n & 64 ? 0  : hi;

        return ret;
    }

    MAKE_BINARY_OP_HELPERS_U64(>>);

    U128& operator<<=(U64 n)
    {
        const U64 hi = __shiftleft128(m_lo, m_hi, (U8)n);
        const U64 lo = m_lo << (n & 63ULL);

        m_hi = n & 64 ? lo : hi;
        m_lo = n & 64 ? 0 : lo;

        return *this;
    }

    friend U128 operator<<(const U128& x, U64 n)
    {
        U128 ret;

        const U64 hi = __shiftleft128(x.m_lo, x.m_hi, (U8)n);
        const U64 lo = x.m_lo << (n & 63ULL);

        ret.m_hi = n & 64 ? lo : hi;
        ret.m_lo = n & 64 ? 0 : lo;

        return ret;
    }

    MAKE_BINARY_OP_HELPERS_U64(<<);

    friend U128 operator~(const U128& x)
    {
        return U128(~x.m_hi, ~x.m_lo);
    }

    friend U128 operator+(const U128& x)
    {
        return x;
    }

    friend U128 operator-(const U128& x)
    {
        U128 ret;
        static_cast<void>(_subborrow_u64(_subborrow_u64(0, 0, x.m_lo, &ret.m_lo), 0, x.m_hi, &ret.m_hi));
        return ret;
    }

    U128& operator++()
    {
        operator+=(1);
        return *this;
    }

    U128 operator++(int)
    {
        const U128 x = *this;
        operator++();
        return x;
    }

    U128& operator--()
    {
        operator-=(1);
        return *this;
    }

    U128 operator--(int)
    {
        const U128 x = *this;
        operator--();
        return x;
    }

    friend bool operator<(const U128& x, const U128& y)
    {
        U64 unusedLo, unusedHi;
        return _subborrow_u64(_subborrow_u64(0, x.m_lo, y.m_lo, &unusedLo), x.m_hi, y.m_hi, &unusedHi);
    }
    MAKE_BINARY_OP_HELPERS(<);
    MAKE_BINARY_OP_HELPERS_FLOAT(<);

    friend bool operator>(const U128& x, const U128& y) { return y < x; }
    MAKE_BINARY_OP_HELPERS(>);
    MAKE_BINARY_OP_HELPERS_FLOAT(>);

    friend bool operator<=(const U128& x, const U128& y) { return !(x > y); }
    MAKE_BINARY_OP_HELPERS(<=);
    MAKE_BINARY_OP_HELPERS_FLOAT(<=);

    friend bool operator>=(const U128& x, const U128& y) { return !(x < y); }
    MAKE_BINARY_OP_HELPERS(>=);
    MAKE_BINARY_OP_HELPERS_FLOAT(>=);

    friend bool operator==(const U128& x, const U128& y)
    {
        return !((x.m_hi ^ y.m_hi) | (x.m_lo ^ y.m_lo));
    }
    MAKE_BINARY_OP_HELPERS(==);
    MAKE_BINARY_OP_HELPERS_FLOAT(==);

    friend bool operator!=(const U128& x, const U128& y) { return !(x == y); }
    MAKE_BINARY_OP_HELPERS(!=);
    MAKE_BINARY_OP_HELPERS_FLOAT(!=);

    explicit operator bool() const { return m_hi | m_lo; }

    operator U8 () const { return (U8) m_lo; }
    operator U16() const { return (U16)m_lo; }
    operator U32() const { return (U32)m_lo; }
    operator U64() const { return (U64)m_lo; }

    operator I8 () const { return (I8) m_lo; }
    operator I16() const { return (I16)m_lo; }
    operator I32() const { return (I32)m_lo; }
    operator I64() const { return (I64)m_lo; }

    operator char() const { return (char)m_lo; }

    // rounding method is implementation-defined as per the Standard [conv.fpint]
    // this implementation performs IEEE 754-compliant "round half to even" rounding to nearest,
    // regardless of the current FPU rounding mode, which matches the behavior of clang and GCC
    operator float() const;

    // rounding method is implementation-defined as per the Standard [conv.fpint]
    // this implementation performs IEEE 754-compliant "round half to even" rounding to nearest,
    // regardless of the current FPU rounding mode, which matches the behavior of clang and GCC
    operator double() const;

    // caller is responsible for ensuring that buf has space for the U128 AND the null terminator
    // that follows, in the given output base.
    // Common bases and worst-case size requirements:
    // Base  2: 129 bytes (128 + null terminator)
    // Base  8:  44 bytes ( 43 + null terminator)
    // Base 10:  40 bytes ( 39 + null terminator)
    // Base 16:  33 bytes ( 32 + null terminator)
    void ToString(char* buf, U64 base = 10) const;

private:
    U64 m_lo;
    U64 m_hi;
};

#undef MAKE_BINARY_OP_HELPERS
#undef MAKE_BINARY_OP_HELPERS_FLOAT
#undef MAKE_BINARY_OP_HELPERS_U64

std::ostream& operator<<(std::ostream& os, const U128& x);
