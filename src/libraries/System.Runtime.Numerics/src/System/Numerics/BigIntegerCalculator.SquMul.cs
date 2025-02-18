// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Buffers;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace System.Numerics
{
    internal static partial class BigIntegerCalculator
    {
#if DEBUG
        // Mutable for unit testing...
        internal static
#else
        internal const
#endif
            int
            MultiplyKaratsubaThreshold = 32,
            MultiplyToom3Threshold = 256;

        public static void Square(ReadOnlySpan<uint> value, Span<uint> bits)
        {
            Debug.Assert(bits.Length == value.Length + value.Length);

            // Executes different algorithms for computing z = a * a
            // based on the actual length of a. If a is "small" enough
            // we stick to the classic "grammar-school" method; for the
            // rest we switch to implementations with less complexity
            // albeit more overhead (which needs to pay off!).

            // NOTE: useful thresholds needs some "empirical" testing,
            // which are smaller in DEBUG mode for testing purpose.

            if (value.Length < MultiplyToom3Threshold)
                SquareKaratsubaOrNaive(value, bits);
            else
                SquareToom3(value, bits);
        }

        private static void SquareToom3(ReadOnlySpan<uint> value, Span<uint> bits)
        {
            Debug.Assert(bits.Length == value.Length + value.Length);

            Debug.Assert(value.Length >= 3);
            Debug.Assert(bits.Length >= value.Length + value.Length);

            // Based on the Toom-Cook multiplication.
            // https://en.wikipedia.org/wiki/Toom-Cook_multiplication

            int n = (value.Length + 2) / 3;
            int pLength = n + 1;
            int pAndQAllLength = pLength * 3;
            uint[]? pAndQAllFromPool = null;
            Span<uint> pAndQAll = (
                (uint)pAndQAllLength <= StackAllocThreshold
                ? stackalloc uint[StackAllocThreshold]
                : pAndQAllFromPool = ArrayPool<uint>.Shared.Rent(pAndQAllLength)).Slice(0, pAndQAllLength);
            pAndQAll.Clear();

            Toom3PData p = Toom3CalcP(value, n, pAndQAll.Slice(0, 3 * pLength));

            // Replace r_n in Wikipedia with z_n
            int rLength = pLength + pLength + 1;
            int rAndZAllLength = rLength * 3;
            uint[]? rAndZAllFromPool = null;
            Span<uint> rAndZAll = (
                (uint)rAndZAllLength <= StackAllocThreshold
                ? stackalloc uint[StackAllocThreshold]
                : rAndZAllFromPool = ArrayPool<uint>.Shared.Rent(rAndZAllLength)).Slice(0, rAndZAllLength);
            rAndZAll.Clear();

            Toom3CalcZSquare(p, n, bits, rAndZAll);

            if (pAndQAllFromPool != null)
                ArrayPool<uint>.Shared.Return(pAndQAllFromPool);

            if (rAndZAllFromPool != null)
                ArrayPool<uint>.Shared.Return(rAndZAllFromPool);
        }
        private static void SquareKaratsubaOrNaive(ReadOnlySpan<uint> value, Span<uint> bits)
        {
            if (value.Length < MultiplyKaratsubaThreshold)
                SquareNaive(value, bits);
            else
                SquareKaratsuba(value, bits);
        }
        private static void SquareKaratsuba(ReadOnlySpan<uint> value, Span<uint> bits)
        {
            Debug.Assert(bits.Length == value.Length + value.Length);

            // Based on the Toom-Cook multiplication we split value
            // into two smaller values, doing recursive squaring.
            // The special form of this multiplication, where we
            // split both operands into two operands, is also known
            // as the Karatsuba algorithm...

            // https://en.wikipedia.org/wiki/Toom-Cook_multiplication
            // https://en.wikipedia.org/wiki/Karatsuba_algorithm

            // Say we want to compute z = a * a ...

            // ... we need to determine our new length (just the half)
            int n = value.Length >> 1;
            int n2 = n << 1;

            // ... split value like a = (a_1 << n) + a_0
            ReadOnlySpan<uint> valueLow = value.Slice(0, n);
            ReadOnlySpan<uint> valueHigh = value.Slice(n);

            // ... prepare our result array (to reuse its memory)
            Span<uint> bitsLow = bits.Slice(0, n2);
            Span<uint> bitsHigh = bits.Slice(n2);

            // ... compute z_0 = a_0 * a_0 (squaring again!)
            SquareKaratsubaOrNaive(valueLow, bitsLow);

            // ... compute z_2 = a_1 * a_1 (squaring again!)
            SquareKaratsubaOrNaive(valueHigh, bitsHigh);

            int foldLength = valueHigh.Length + 1;
            uint[]? foldFromPool = null;
            Span<uint> fold = ((uint)foldLength <= StackAllocThreshold ?
                              stackalloc uint[StackAllocThreshold]
                              : foldFromPool = ArrayPool<uint>.Shared.Rent(foldLength)).Slice(0, foldLength);
            fold.Clear();

            int coreLength = foldLength + foldLength;
            uint[]? coreFromPool = null;
            Span<uint> core = ((uint)coreLength <= StackAllocThreshold ?
                              stackalloc uint[StackAllocThreshold]
                              : coreFromPool = ArrayPool<uint>.Shared.Rent(coreLength)).Slice(0, coreLength);
            core.Clear();

            // ... compute z_a = a_1 + a_0 (call it fold...)
            Add(valueHigh, valueLow, fold);

            // ... compute z_1 = z_a * z_a - z_0 - z_2
            SquareKaratsubaOrNaive(fold, core);

            if (foldFromPool != null)
                ArrayPool<uint>.Shared.Return(foldFromPool);

            SubtractCore(bitsHigh, bitsLow, core);

            // ... and finally merge the result! :-)
            AddSelf(bits.Slice(n), core);

            if (coreFromPool != null)
                ArrayPool<uint>.Shared.Return(coreFromPool);
        }

        private static void SquareNaive(ReadOnlySpan<uint> value, Span<uint> bits)
        {
            Debug.Assert(bits.Length == value.Length + value.Length);

            // Switching to managed references helps eliminating
            // index bounds check...
            ref uint resultPtr = ref MemoryMarshal.GetReference(bits);

            // Squares the bits using the "grammar-school" method.
            // Envisioning the "rhombus" of a pen-and-paper calculation
            // we see that computing z_i+j += a_j * a_i can be optimized
            // since a_j * a_i = a_i * a_j (we're squaring after all!).
            // Thus, we directly get z_i+j += 2 * a_j * a_i + c.

            // ATTENTION: an ordinary multiplication is safe, because
            // z_i+j + a_j * a_i + c <= 2(2^32 - 1) + (2^32 - 1)^2 =
            // = 2^64 - 1 (which perfectly matches with ulong!). But
            // here we would need an UInt65... Hence, we split these
            // operation and do some extra shifts.
            for (int i = 0; i < value.Length; i++)
            {
                ulong carry = 0UL;
                uint v = value[i];
                for (int j = 0; j < i; j++)
                {
                    ulong digit1 = Unsafe.Add(ref resultPtr, i + j) + carry;
                    ulong digit2 = (ulong)value[j] * v;
                    Unsafe.Add(ref resultPtr, i + j) = unchecked((uint)(digit1 + (digit2 << 1)));
                    carry = (digit2 + (digit1 >> 1)) >> 31;
                }
                ulong digits = (ulong)v * v + carry;
                Unsafe.Add(ref resultPtr, i + i) = unchecked((uint)digits);
                Unsafe.Add(ref resultPtr, i + i + 1) = (uint)(digits >> 32);
            }
        }

        public static void Multiply(ReadOnlySpan<uint> left, uint right, Span<uint> bits)
        {
            Debug.Assert(bits.Length == left.Length + 1);

            // Executes the multiplication for one big and one 32-bit integer.
            // Since every step holds the already slightly familiar equation
            // a_i * b + c <= 2^32 - 1 + (2^32 - 1)^2 < 2^64 - 1,
            // we are safe regarding to overflows.

            int i = 0;
            ulong carry = 0UL;

            for (; i < left.Length; i++)
            {
                ulong digits = (ulong)left[i] * right + carry;
                bits[i] = unchecked((uint)digits);
                carry = digits >> 32;
            }
            bits[i] = (uint)carry;
        }

        /// <summary>
        /// A wrapper of <see cref="Multiply(ReadOnlySpan{uint}, ReadOnlySpan{uint}, Span{uint})"/>.
        /// </summary>
        /// /// <remarks>
        /// The order of <paramref name="left"/> and <paramref name="right"/> does not matter.
        /// The method internally swaps them if necessary to ensure correct computation.
        /// </remarks>
        /// <returns>The length of <paramref name="bits"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int MultiplySafe(ReadOnlySpan<uint> left, ReadOnlySpan<uint> right, Span<uint> bits)
        {
            if (left.Length < right.Length)
            {
                ReadOnlySpan<uint> tmp = left;
                left = right;
                right = tmp;
            }

            if (right.Length != 0)
            {
                Multiply(left, right, bits);

                int len = left.Length + right.Length;
                if (bits[len - 1] == 0)
                    --len;

                return len;
            }
            return 0;
        }

        public static void Multiply(ReadOnlySpan<uint> left, ReadOnlySpan<uint> right, Span<uint> bits)
        {
            Debug.Assert(left.Length >= right.Length);
            Debug.Assert(right.Length >= 0);
            Debug.Assert(left[^1] != 0);
            Debug.Assert(right[^1] != 0);
            Debug.Assert(bits.Length >= left.Length + right.Length);
            Debug.Assert(bits.Trim(0u).IsEmpty);
            Debug.Assert(MultiplyKaratsubaThreshold >= 2);
            Debug.Assert(MultiplyToom3Threshold >= 9);
            Debug.Assert(MultiplyKaratsubaThreshold <= MultiplyToom3Threshold);

            // Executes different algorithms for computing z = a * b
            // based on the actual length of b. If b is "small" enough
            // we stick to the classic "grammar-school" method; for the
            // rest we switch to implementations with less complexity
            // albeit more overhead (which needs to pay off!).

            // NOTE: useful thresholds needs some "empirical" testing,
            // which are smaller in DEBUG mode for testing purpose.

            if (right.Length < MultiplyToom3Threshold)
                MultiplyKaratsubaOrNaive(left, right, bits);
            else
                MultiplyToom3(left, right, bits);
        }

        private static void MultiplyKaratsubaOrNaive(ReadOnlySpan<uint> left, ReadOnlySpan<uint> right, Span<uint> bits)
        {
            Debug.Assert(left.Length >= right.Length);
            Debug.Assert(bits.Length >= left.Length + right.Length);
            Debug.Assert(!bits.ContainsAnyExcept(0u));

            if (right.Length < MultiplyKaratsubaThreshold)
            {
                MultiplyNaive(left, right, bits);
                return;
            }

            //                                            upper           lower
            // A=   |               |               | a1 = a[n..2n] | a0 = a[0..n] |
            // B=   |               |               | b1 = b[n..2n] | b0 = b[0..n] |

            // Result
            // z0=  |               |               |            a0 * b0            |
            // z1=  |               |       a1 * b0 + a0 * b1       |               |
            // z2=  |            a1 * b1            |               |               |

            // z1 = a1 * b0 + a0 * b1
            //    = (a0 + a1) * (b0 + b1) - a0 * b0 - a1 * b1
            //    = (a0 + a1) * (b0 + b1) - z0 - z2


            // Based on the Toom-Cook multiplication we split left/right
            // into two smaller values, doing recursive multiplication.
            // The special form of this multiplication, where we
            // split both operands into two operands, is also known
            // as the Karatsuba algorithm...

            // https://en.wikipedia.org/wiki/Toom-Cook_multiplication
            // https://en.wikipedia.org/wiki/Karatsuba_algorithm

            // Say we want to compute z = a * b ...

            // ... we need to determine our new length (just the half)
            int n = (left.Length + 1) >> 1;
            if (right.Length > n)
                MultiplyKaratsuba(left, right, bits, n);
            else
                MultiplyKaratsubaRightSmall(left, right, bits, n);
        }

        private static void MultiplyKaratsubaRightSmall(ReadOnlySpan<uint> left, ReadOnlySpan<uint> right, Span<uint> bits, int n)
        {
            Debug.Assert(left.Length >= right.Length);
            Debug.Assert(2 * n - left.Length is 0 or 1);
            Debug.Assert(right.Length <= n);
            Debug.Assert(bits.Length >= left.Length + right.Length);

            // ... split left like a = (a_1 << n) + a_0
            ReadOnlySpan<uint> leftLow = left.Slice(0, n);
            ReadOnlySpan<uint> leftHigh = left.Slice(n);
            Debug.Assert(leftLow.Length >= leftHigh.Length);

            // ... prepare our result array (to reuse its memory)
            Span<uint> bitsLow = bits.Slice(0, n + right.Length);
            Span<uint> bitsHigh = bits.Slice(n);

            // ... compute low
            MultiplyKaratsubaOrNaive(leftLow, right, bitsLow);

            int carryLength = right.Length;
            uint[]? carryFromPool = null;
            Span<uint> carry = ((uint)carryLength <= StackAllocThreshold ?
                              stackalloc uint[StackAllocThreshold]
                              : carryFromPool = ArrayPool<uint>.Shared.Rent(carryLength)).Slice(0, carryLength);

            Span<uint> carryOrig = bits.Slice(n, right.Length);
            carryOrig.CopyTo(carry);
            carryOrig.Clear();

            // ... compute high
            if (leftHigh.Length < right.Length)
                MultiplyKaratsubaOrNaive(right, leftHigh, bitsHigh.Slice(0, leftHigh.Length + right.Length));
            else
                MultiplyKaratsubaOrNaive(leftHigh, right, bitsHigh.Slice(0, leftHigh.Length + right.Length));

            AddSelf(bitsHigh, carry);

            if (carryFromPool != null)
                ArrayPool<uint>.Shared.Return(carryFromPool);
        }

        private static void MultiplyKaratsuba(ReadOnlySpan<uint> left, ReadOnlySpan<uint> right, Span<uint> bits, int n)
        {
            Debug.Assert(left.Length >= right.Length);
            Debug.Assert(2 * n - left.Length is 0 or 1);
            Debug.Assert(right.Length > n);
            Debug.Assert(bits.Length >= left.Length + right.Length);

            if (right.Length < MultiplyKaratsubaThreshold)
            {
                MultiplyNaive(left, right, bits);
                return;
            }

            // ... split left like a = (a_1 << n) + a_0
            ReadOnlySpan<uint> leftLow = left.Slice(0, n);
            ReadOnlySpan<uint> leftHigh = left.Slice(n);

            // ... split right like b = (b_1 << n) + b_0
            ReadOnlySpan<uint> rightLow = right.Slice(0, n);
            ReadOnlySpan<uint> rightHigh = right.Slice(n);

            // ... prepare our result array (to reuse its memory)
            Span<uint> bitsLow = bits.Slice(0, n + n);
            Span<uint> bitsHigh = bits.Slice(n + n);

            Debug.Assert(leftLow.Length >= leftHigh.Length);
            Debug.Assert(rightLow.Length >= rightHigh.Length);
            Debug.Assert(bitsLow.Length >= bitsHigh.Length);

            // ... compute z_0 = a_0 * b_0 (multiply again)
            MultiplyKaratsuba(leftLow, rightLow, bitsLow, (leftLow.Length + 1) >> 1);

            // ... compute z_2 = a_1 * b_1 (multiply again)
            MultiplyKaratsubaOrNaive(leftHigh, rightHigh, bitsHigh);

            int foldLength = n + 1;
            uint[]? leftFoldFromPool = null;
            Span<uint> leftFold = ((uint)foldLength <= StackAllocThreshold ?
                                  stackalloc uint[StackAllocThreshold]
                                  : leftFoldFromPool = ArrayPool<uint>.Shared.Rent(foldLength)).Slice(0, foldLength);
            leftFold.Clear();

            uint[]? rightFoldFromPool = null;
            Span<uint> rightFold = ((uint)foldLength <= StackAllocThreshold ?
                                   stackalloc uint[StackAllocThreshold]
                                   : rightFoldFromPool = ArrayPool<uint>.Shared.Rent(foldLength)).Slice(0, foldLength);
            rightFold.Clear();

            // ... compute z_a = a_1 + a_0 (call it fold...)
            Add(leftLow, leftHigh, leftFold);

            // ... compute z_b = b_1 + b_0 (call it fold...)
            Add(rightLow, rightHigh, rightFold);

            int coreLength = foldLength + foldLength;
            uint[]? coreFromPool = null;
            Span<uint> core = ((uint)coreLength <= StackAllocThreshold ?
                              stackalloc uint[StackAllocThreshold]
                              : coreFromPool = ArrayPool<uint>.Shared.Rent(coreLength)).Slice(0, coreLength);
            core.Clear();

            // ... compute z_ab = z_a * z_b
            MultiplyKaratsuba(leftFold, rightFold, core, (leftFold.Length + 1) >> 1);

            if (leftFoldFromPool != null)
                ArrayPool<uint>.Shared.Return(leftFoldFromPool);

            if (rightFoldFromPool != null)
                ArrayPool<uint>.Shared.Return(rightFoldFromPool);

            // ... compute z_1 = z_a * z_b - z_0 - z_2 = a_0 * b_1 + a_1 * b_0
            SubtractCore(bitsLow, bitsHigh, core);

            Debug.Assert(ActualLength(core) <= left.Length + 1);

            // ... and finally merge the result! :-)
            AddSelf(bits.Slice(n), TrimEnd(core));

            if (coreFromPool != null)
                ArrayPool<uint>.Shared.Return(coreFromPool);
        }

        private static void MultiplyToom3(ReadOnlySpan<uint> left, ReadOnlySpan<uint> right, Span<uint> bits)
        {
            Debug.Assert(left.Length >= 3);
            Debug.Assert(left.Length >= right.Length);
            Debug.Assert(bits.Length >= left.Length + right.Length);

            // Based on the Toom-Cook multiplication.
            // Replace m in Wikipedia with left and n in Wikipedia with right.
            // https://en.wikipedia.org/wiki/Toom-Cook_multiplication

            int n = (left.Length + 2) / 3;
            if (right.Length <= n)
            {
                MultiplyToom3OneThird(left, right, bits, n);
                return;
            }

            int pLength = n + 1;
            int pAndQAllLength = pLength * 6;
            uint[]? pAndQAllFromPool = null;
            Span<uint> pAndQAll = (
                (uint)pAndQAllLength <= StackAllocThreshold
                ? stackalloc uint[StackAllocThreshold]
                : pAndQAllFromPool = ArrayPool<uint>.Shared.Rent(pAndQAllLength)).Slice(0, pAndQAllLength);
            pAndQAll.Clear();

            Toom3PData p = Toom3CalcP(left, n, pAndQAll.Slice(0, 3 * pLength));
            Toom3PData q = Toom3CalcP(right, n, pAndQAll.Slice(3 * pLength));

            // Replace r_n in Wikipedia with z_n
            int rLength = pLength + pLength + 1;
            int rAndZAllLength = rLength * 3;
            uint[]? rAndZAllFromPool = null;
            Span<uint> rAndZAll = (
                (uint)rAndZAllLength <= StackAllocThreshold
                ? stackalloc uint[StackAllocThreshold]
                : rAndZAllFromPool = ArrayPool<uint>.Shared.Rent(rAndZAllLength)).Slice(0, rAndZAllLength);
            rAndZAll.Clear();

            Toom3CalcZ(p, q, n, bits, rAndZAll);

            if (pAndQAllFromPool != null)
                ArrayPool<uint>.Shared.Return(pAndQAllFromPool);

            if (rAndZAllFromPool != null)
                ArrayPool<uint>.Shared.Return(rAndZAllFromPool);
        }

        private static void MultiplyToom3OneThird(ReadOnlySpan<uint> left, ReadOnlySpan<uint> right, Span<uint> bits, int n)
        {

            // A=   |               | a2 = a[2n..3n]| a1 = a[n..2n] |  a0 = a[0..n] |
            // B=   |               |               |               |       b       |

            // Result
            // z0=  |               |               |            a0 * b             |
            // z1=  |               |            a1 * b             |               |
            // z2=  |            a2 * b             |               |               |


            Debug.Assert(3 * n - 2 <= left.Length);
            Debug.Assert(left.Length <= 3 * n);
            Debug.Assert(right.Length <= n);

            ReadOnlySpan<uint> left0 = left.Slice(0, n);
            ReadOnlySpan<uint> left1 = left.Slice(n, n);
            ReadOnlySpan<uint> left2 = left.Slice(n + n);

            // ... compute high
            MultiplySafe(left2, right, bits.Slice(n + n));

            // ... compute low
            MultiplyToom3(left0, right, bits);

            int coreLength = n + right.Length;
            uint[]? coreFromPool = null;
            Span<uint> core = (
                (uint)coreLength <= StackAllocThreshold
                ? stackalloc uint[StackAllocThreshold]
                : coreFromPool = ArrayPool<uint>.Shared.Rent(coreLength)).Slice(0, coreLength);
            core.Clear();

            // ... compute mid
            MultiplyToom3(left1, right, core);

            AddSelf(bits.Slice(n), core);

            if (coreFromPool != null)
                ArrayPool<uint>.Shared.Return(coreFromPool);
        }

        private readonly ref struct Toom3PData(
            ReadOnlySpan<uint> p0,
            ReadOnlySpan<uint> pInf,
            ReadOnlySpan<uint> p1,
            ReadOnlySpan<uint> pm1,
            int pm1Sign,
            ReadOnlySpan<uint> pm2,
            int pm2Sign)
        {
            public readonly ReadOnlySpan<uint> p0 = p0;
            public readonly ReadOnlySpan<uint> p1 = p1;
            public readonly ReadOnlySpan<uint> pInf = pInf;
            public readonly ReadOnlySpan<uint> pm1 = pm1;
            public readonly ReadOnlySpan<uint> pm2 = pm2;
            public readonly int pm1Sign = pm1Sign;
            public readonly int pm2Sign = pm2Sign;
        }
        private static Toom3PData Toom3CalcP(ReadOnlySpan<uint> value, int n, Span<uint> buffer)
        {
            Debug.Assert(!buffer.ContainsAnyExcept(0u));
            Debug.Assert(buffer.Length == 3 * (n + 1));
            Debug.Assert(value.Length > n);
            Debug.Assert(value[^1] != 0);

            int pLength = n + 1;

            ReadOnlySpan<uint> value0, value1, value2;

            value0 = TrimEnd(value.Slice(0, n));
            if (value.Length <= n + n)
            {
                value1 = value.Slice(n);
                value2 = default;
            }
            else
            {
                value1 = TrimEnd(value.Slice(n, n));
                value2 = value.Slice(n + n);
            }

            Span<uint> p1 = buffer.Slice(0, pLength);
            Span<uint> pm1 = buffer.Slice(pLength, pLength);
            Toom3CalcP1AndPm1(value0, value1, value2, p1, pm1, out int pm1Sign);

            pm1 = pm1Sign != 0 ? TrimEnd(pm1) : default;

            ReadOnlySpan<uint> pm2 = Toom3CalcPm2(pm1, pm1Sign, value0, value2, buffer.Slice(pLength + pLength, pLength), out int pm2Sign);

            return new Toom3PData(
                p0: value0,
                p1: TrimEnd(p1),
                pInf: value2,
                pm1: TrimEnd(pm1),
                pm2: pm2,
                pm1Sign: pm1Sign,
                pm2Sign: pm2Sign
            );

            // Calculate p(1) = p_0 + m_1, p(-1) = p_0 - m_1
            static void Toom3CalcP1AndPm1(ReadOnlySpan<uint> v0, ReadOnlySpan<uint> v1, ReadOnlySpan<uint> v2, Span<uint> p1, Span<uint> pm1, out int pm1Sign)
            {
                Debug.Assert(p1.Length >= Math.Max(Math.Max(v0.Length, v1.Length), v2.Length));
                Debug.Assert(pm1.Length >= Math.Max(Math.Max(v0.Length, v1.Length), v2.Length));
                Debug.Assert(v1.IsEmpty || v1[^1] != 0);

                v0.CopyTo(p1);
                AddSelf(p1, v2);

                p1.CopyTo(pm1);

                AddSelf(p1, v1);

                pm1Sign = 1;
                SubtractSelf(pm1, ref pm1Sign, v1);
            }

            // Calculate p(-2) = (p(-1) + m_2)*2 - m_0
            static ReadOnlySpan<uint> Toom3CalcPm2(ReadOnlySpan<uint> pm1, int pm1Sign, ReadOnlySpan<uint> v0, ReadOnlySpan<uint> v2, Span<uint> span, out int sign)
            {
                Debug.Assert(!span.ContainsAnyExcept(0u));
                Debug.Assert(pm1.IsEmpty || pm1[^1] != 0);
                Debug.Assert(v0.IsEmpty || v0[^1] != 0);
                Debug.Assert(v2.IsEmpty || v2[^1] != 0);

                pm1.CopyTo(span);
                sign = pm1Sign;

                // Calclate p(-1) + m_2
                AddSelf(span, ref sign, v2);

                // Calculate p(-2) = (p(-1) + m_2)*2
                {
                    Debug.Assert(span[^1] < 0x8000_0000);
                    uint shiftCarry = 0;
                    LeftShiftSelf(span, 1, ref shiftCarry);
                }

                Debug.Assert(span[^1] != uint.MaxValue);

                // Calculate p(-2) = (p(-1) + m_2)*2 - m_0
                SubtractSelf(span, ref sign, v0);

                return TrimEnd(span);
            }
        }

        private static void Toom3CalcZ(Toom3PData left, Toom3PData right, int n, Span<uint> bits, Span<uint> buffer)
        {
            Debug.Assert(!buffer.ContainsAnyExcept(0u));
            Debug.Assert(left.pInf.Length >= right.pInf.Length);

            int rLength = n + n + 3;

            ReadOnlySpan<uint> p0 = left.p0;
            ReadOnlySpan<uint> q0 = right.p0;

            ReadOnlySpan<uint> p1 = left.p1;
            ReadOnlySpan<uint> q1 = right.p1;

            ReadOnlySpan<uint> pm1 = left.pm1;
            ReadOnlySpan<uint> qm1 = right.pm1;

            ReadOnlySpan<uint> pm2 = left.pm2;
            ReadOnlySpan<uint> qm2 = right.pm2;

            ReadOnlySpan<uint> pInf = left.pInf;
            ReadOnlySpan<uint> qInf = right.pInf;


            Span<uint> r0 = bits.Slice(0, p0.Length + q0.Length);
            Span<uint> rInf =
                !qInf.IsEmpty
                ? bits.Slice(4 * n, pInf.Length + qInf.Length)
                : default;

            Span<uint> r1 = buffer.Slice(0, p1.Length + q1.Length);
            Span<uint> rm1 = buffer.Slice(rLength, pm1.Length + qm1.Length);
            Span<uint> rm2 = buffer.Slice(rLength * 2, pm2.Length + qm2.Length);

            r0 = MultiplySafe(p0, q0, r0) switch { var len => r0.Slice(0, len) };
            r1 = MultiplySafe(p1, q1, r1) switch { var len => r1.Slice(0, len) };
            rm1 = MultiplySafe(pm1, qm1, rm1) switch { var len => rm1.Slice(0, len) };
            MultiplySafe(pm2, qm2, rm2);
            rInf = MultiplySafe(pInf, qInf, rInf) switch { var len => rInf.Slice(0, len) };

            Toom3CalcResult(
                n,
                r0: TrimEnd(r0),
                rInf: TrimEnd(rInf),
                z1: buffer.Slice(0, rLength),
                r1Length: ActualLength(r1),
                z2: buffer.Slice(rLength, rLength),
                z2Sign: left.pm1Sign * right.pm1Sign,
                rm1Length: ActualLength(rm1),
                z3: buffer.Slice(rLength * 2, rLength),
                z3Sign: left.pm2Sign * right.pm2Sign,
                bits
            );
        }
        private static void Toom3CalcZSquare(Toom3PData value, int n, Span<uint> bits, Span<uint> buffer)
        {
            Debug.Assert(!buffer.ContainsAnyExcept(0u));
            Debug.Assert(!value.pInf.IsEmpty);

            int rLength = n + n + 3;

            ReadOnlySpan<uint> p0 = value.p0;
            ReadOnlySpan<uint> p1 = value.p1;
            ReadOnlySpan<uint> pm1 = value.pm1;
            ReadOnlySpan<uint> pm2 = value.pm2;
            ReadOnlySpan<uint> pInf = value.pInf;

            Span<uint> r0 = bits.Slice(0, p0.Length << 1);
            Span<uint> rInf = bits.Slice(4 * n, pInf.Length << 1);

            Span<uint> r1 = buffer.Slice(0, p1.Length << 1);
            Span<uint> rm1 = buffer.Slice(rLength, pm1.Length << 1);
            Span<uint> rm2 = buffer.Slice(rLength * 2, pm2.Length << 1);

            Square(p0, r0);
            Square(p1, r1);
            Square(pm1, rm1);
            Square(pm2, rm2);
            Square(pInf, rInf);

            Toom3CalcResult(
                n,
                r0: TrimEnd(r0),
                rInf: TrimEnd(rInf),
                z1: buffer.Slice(0, rLength),
                r1Length: ActualLength(r1),
                z2: buffer.Slice(rLength, rLength),
                z2Sign: value.pm1Sign & 1,
                rm1Length: ActualLength(rm1),
                z3: buffer.Slice(rLength * 2, rLength),
                z3Sign: value.pm2Sign & 1,
                bits
            );
        }

        private static void Toom3CalcResult(
            int n,
            ReadOnlySpan<uint> r0,
            ReadOnlySpan<uint> rInf,
            Span<uint> z1,
            int r1Length,
            Span<uint> z2,
            int z2Sign,
            int rm1Length,
            Span<uint> z3,
            int z3Sign,
            Span<uint> bits)
        {
            int z1Sign = Math.Sign(r1Length);

            // Calc z_3 = (r(-2) - r(1))/3
            {
                // Calc r(-2) - r(1)
                SubtractSelf(z3, ref z3Sign, z1.Slice(0, r1Length));

                // Calc (r(-2) - r(1))/3
                DivideThreeSelf(TrimEnd(z3));
            }

            // Calc z_1 = (r(1) - r(-1))/2
            {
                SubtractSelf(z1, ref z1Sign, z2.Slice(0, rm1Length), z2Sign);
                Debug.Assert(z1.IsEmpty || (z1[0] & 1) == 0);

                uint z1Carry = 0;
                RightShiftSelf(z1, 1, ref z1Carry);
            }

            // Calc z_2 = r(-1) - r(0)
            SubtractSelf(z2, ref z2Sign, r0);

            // Calc z_3 = (z_2 - z_3)/2 + 2r(Inf)
            {
                // Calc z_2 - z_3
                SubtractSelf(z3, ref z3Sign, z2, z2Sign);
                z3Sign = -z3Sign;

                Debug.Assert(z3.IsEmpty || (z3[0] & 1) == 0);


                // Calc (z_2 - z_3)/2
                uint z3Carry = 0;
                RightShiftSelf(z3, 1, ref z3Carry);

                // Calc (z_2 - z_3)/2 + 2r(Inf)
                AddSelf(z3, ref z3Sign, rInf);
                AddSelf(z3, ref z3Sign, rInf);
            }

            // Calc z_2 = z_2 + z_1 - r(Inf)
            {
                AddSelf(z2, ref z2Sign, TrimEnd(z1));
                SubtractSelf(z2, ref z2Sign, rInf);
            }

            // Calc z_1 = z_1 - z_3
            SubtractSelf(z1, ref z1Sign, TrimEnd(z3));

            Debug.Assert(z1Sign >= 0);
            Debug.Assert(z2Sign >= 0);
            Debug.Assert(z3Sign >= 0);

            AddSelf(bits.Slice(n), TrimEnd(z1));
            AddSelf(bits.Slice(2 * n), TrimEnd(z2));

            if (bits.Length >= 3 * n)
                AddSelf(bits.Slice(3 * n), TrimEnd(z3));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void MultiplyNaive(ReadOnlySpan<uint> left, ReadOnlySpan<uint> right, Span<uint> bits)
        {
            Debug.Assert(right.Length < MultiplyKaratsubaThreshold);

            // Switching to managed references helps eliminating
            // index bounds check...
            ref uint resultPtr = ref MemoryMarshal.GetReference(bits);

            // Multiplies the bits using the "grammar-school" method.
            // Envisioning the "rhombus" of a pen-and-paper calculation
            // should help getting the idea of these two loops...
            // The inner multiplication operations are safe, because
            // z_i+j + a_j * b_i + c <= 2(2^32 - 1) + (2^32 - 1)^2 =
            // = 2^64 - 1 (which perfectly matches with ulong!).

            for (int i = 0; i < right.Length; i++)
            {
                uint rv = right[i];
                ulong carry = 0UL;
                for (int j = 0; j < left.Length; j++)
                {
                    ref uint elementPtr = ref Unsafe.Add(ref resultPtr, i + j);
                    ulong digits = elementPtr + carry + (ulong)left[j] * rv;
                    elementPtr = unchecked((uint)digits);
                    carry = digits >> 32;
                }
                Unsafe.Add(ref resultPtr, i + left.Length) = (uint)carry;
            }
        }

        private static void DivideThreeSelf(Span<uint> bits)
        {
            const uint oneThird = (uint)((1ul << 32) / 3);
            const uint twoThirds = (uint)((2ul << 32) / 3);

            uint carry = 0;
            for (int i = bits.Length - 1; i >= 0; i--)
            {
                (uint quo, uint rem) = Math.DivRem(bits[i], 3);

                Debug.Assert(carry < 3);

                if (carry == 0)
                {
                    bits[i] = quo;
                    carry = rem;
                }
                else if (carry == 1)
                {
                    if (++rem == 3)
                    {
                        rem = 0;
                        ++quo;
                    }

                    bits[i] = oneThird + quo;
                    carry = rem;
                }
                else
                {
                    if ((rem += 2) >= 3)
                    {
                        rem -= 3;
                        ++quo;
                    }

                    Debug.Assert(quo <= uint.MaxValue - twoThirds);

                    bits[i] = twoThirds + quo;
                    carry = rem;
                }
            }

            Debug.Assert(carry == 0);
        }
        private static void SubtractCore(ReadOnlySpan<uint> left, ReadOnlySpan<uint> right, Span<uint> core)
        {
            Debug.Assert(left.Length >= right.Length);
            Debug.Assert(core.Length >= left.Length);

            // Executes a special subtraction algorithm for the multiplication,
            // which needs to subtract two different values from a core value,
            // while core is always bigger than the sum of these values.

            // NOTE: we could do an ordinary subtraction of course, but we spare
            // one "run", if we do this computation within a single one...

            int i = 0;
            long carry = 0L;

            // Switching to managed references helps eliminating
            // index bounds check...
            ref uint leftPtr = ref MemoryMarshal.GetReference(left);
            ref uint corePtr = ref MemoryMarshal.GetReference(core);

            for (; i < right.Length; i++)
            {
                long digit = (Unsafe.Add(ref corePtr, i) + carry) - Unsafe.Add(ref leftPtr, i) - right[i];
                Unsafe.Add(ref corePtr, i) = unchecked((uint)digit);
                carry = digit >> 32;
            }

            for (; i < left.Length; i++)
            {
                long digit = (Unsafe.Add(ref corePtr, i) + carry) - left[i];
                Unsafe.Add(ref corePtr, i) = unchecked((uint)digit);
                carry = digit >> 32;
            }

            for (; carry != 0 && i < core.Length; i++)
            {
                long digit = core[i] + carry;
                core[i] = (uint)digit;
                carry = digit >> 32;
            }
        }
    }
}
