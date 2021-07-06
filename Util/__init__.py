import struct


# adapted from https://stackoverflow.com/questions/14431170/get-the-bits-of-a-float-in-python
def float_to_bits(d):
    s = struct.pack('>d', d)
    return struct.unpack('>q', s)[0]


def bits_to_float(b):
    s = struct.pack('>q', b)
    return struct.unpack('>d', s)[0]


def float_to_binary(d):
    """
    Given a double, returns the bit representation of the input as a string.
    Useful for debugging
    :param d: double
    :return: string showing bit representation of the input
    """
    return bin(float_to_bits(d))[2:].zfill(64)
