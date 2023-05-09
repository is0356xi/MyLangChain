import struct

binary = b'\x81]\x00\x00'
value_little, = struct.unpack("<i", binary)
value_big, = struct.unpack(">i", binary)

print(value_little, value_big)