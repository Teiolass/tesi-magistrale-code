all: generator.so

python.zig:
	zig translate-c /usr/include/python3.11/Python.h -I/usr/include/python3.11 -I/usr/include > python.zig

generator.so: main.zig out.zig numpy_data.zig python.zig
	zig build-lib -I/usr/include -I/usr/include/python3.11 -dynamic -fPIC --name generator -lc main.zig
	mv libgenerator.so generator.so
	rm libgenerator.so.o

release: main.zig out.zig numpy_data.zig python.zig
	zig build-lib -I/usr/include -I/usr/include/python3.11 -dynamic -fPIC --name generator -Doptimize=ReleaseFast main.zig
	mv libgenerator.so generator.so
	rm libgenerator.so.o

