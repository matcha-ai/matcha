.PHONY: build clean

build:
	@mkdir -p build && cd build && cmake .. && cmake --build .

clean:
	@rm -r build*/

