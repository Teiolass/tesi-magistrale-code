all: out.zig generator.so

out.zig:
	zig translate-c /usr/include/python3.11/Python.h -I/usr/include/python3.11 -I/usr/include > out.zig

generator.so: main.zig out.zig numpy_data.zig
	zig build-lib -I/usr/include -I/usr/include/python3.11 -dynamic -fPIC --name test main.zig
	mv libtest.so generator.so
	rm libtest.so.o


