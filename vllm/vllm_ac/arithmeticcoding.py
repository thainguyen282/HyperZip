"""Reference arithmetic coder adapted for integer cumulative-frequency arrays.

Copyright (c) Project Nayuki. MIT license: see ARITHMETIC_CODING_LICENSE.txt.
https://www.nayuki.io/page/reference-arithmetic-coding

Derived from the project's existing adaptation of the Nayuki reference coder.
Unused commented frequency-table code and debug statements have been omitted.
"""

class ArithmeticCoderBase(object):

    def __init__(self, statesize):
        self.STATE_SIZE = statesize
        self.MAX_RANGE = 1 << self.STATE_SIZE
        self.MIN_RANGE = (self.MAX_RANGE >> 2) + 2
        self.MAX_TOTAL = self.MIN_RANGE
        self.MASK = self.MAX_RANGE - 1
        self.TOP_MASK = self.MAX_RANGE >> 1
        self.SECOND_MASK = self.TOP_MASK >> 1
        self.low = 0
        self.high = self.MASK

    def update(self, cumul, symbol):
        low = self.low
        high = self.high
        range = high - low + 1
        total = cumul[-1].item()
        symlow = cumul[symbol].item()
        symhigh = cumul[symbol + 1].item()
        newlow = low + symlow * range // total
        newhigh = low + symhigh * range // total - 1
        self.low = newlow
        self.high = newhigh
        while (self.low ^ self.high) & self.TOP_MASK == 0:
            self.shift()
            self.low = self.low << 1 & self.MASK
            self.high = self.high << 1 & self.MASK | 1
        while self.low & ~self.high & self.SECOND_MASK != 0:
            self.underflow()
            self.low = self.low << 1 & self.MASK >> 1
            self.high = self.high << 1 & self.MASK >> 1 | self.TOP_MASK | 1

    def shift(self):
        raise NotImplementedError()

    def underflow(self):
        raise NotImplementedError()


class ArithmeticEncoder(ArithmeticCoderBase):

    def __init__(self, statesize, bitout):
        super(ArithmeticEncoder, self).__init__(statesize)
        self.output = bitout
        self.num_underflow = 0

    def write(self, cumul, symbol):
        self.update(cumul, symbol)

    def finish(self):
        self.output.write(1)

    def shift(self):
        bit = self.low >> self.STATE_SIZE - 1
        self.output.write(bit)
        for _ in range(self.num_underflow):
            self.output.write(bit ^ 1)
        self.num_underflow = 0

    def underflow(self):
        self.num_underflow += 1


class ArithmeticDecoder(ArithmeticCoderBase):

    def __init__(self, statesize, bitin):
        super(ArithmeticDecoder, self).__init__(statesize)
        self.input = bitin
        self.code = 0
        for _ in range(self.STATE_SIZE):
            self.code = self.code << 1 | self.read_code_bit()

    def read(self, cumul, alphabet_size):
        total = cumul[-1].item()
        range = self.high - self.low + 1
        offset = self.code - self.low
        value = ((offset + 1) * total - 1) // range
        start = 0
        end = alphabet_size
        while end - start > 1:
            middle = start + end >> 1
            if cumul[middle] > value:
                end = middle
            else:
                start = middle
        symbol = start
        self.update(cumul, symbol)
        return symbol

    def shift(self):
        self.code = self.code << 1 & self.MASK | self.read_code_bit()

    def underflow(self):
        self.code = self.code & self.TOP_MASK | self.code << 1 & self.MASK >> 1 | self.read_code_bit()

    def read_code_bit(self):
        temp = self.input.read()
        if temp == -1:
            temp = 0
        return temp


class BitInputStream(object):

    def __init__(self, inp):
        self.input = inp
        self.currentbyte = 0
        self.numbitsremaining = 0

    def read(self):
        if self.currentbyte == -1:
            return -1
        if self.numbitsremaining == 0:
            temp = self.input.read(1)
            if len(temp) == 0:
                self.currentbyte = -1
                return -1
            self.currentbyte = temp[0]
            self.numbitsremaining = 8
        assert self.numbitsremaining > 0
        self.numbitsremaining -= 1
        return self.currentbyte >> self.numbitsremaining & 1

    def read_no_eof(self):
        result = self.read()
        if result != -1:
            return result
        else:
            raise EOFError()

    def close(self):
        self.input.close()
        self.currentbyte = -1
        self.numbitsremaining = 0


class BitOutputStream(object):

    def __init__(self, out):
        self.output = out
        self.currentbyte = 0
        self.numbitsfilled = 0

    def write(self, b):
        if b not in (0, 1):
            raise ValueError('Argument must be 0 or 1')
        self.currentbyte = self.currentbyte << 1 | b
        self.numbitsfilled += 1
        if self.numbitsfilled == 8:
            towrite = bytes((self.currentbyte,))
            self.output.write(towrite)
            self.currentbyte = 0
            self.numbitsfilled = 0

    def close(self):
        while self.numbitsfilled != 0:
            self.write(0)
        self.output.close()
