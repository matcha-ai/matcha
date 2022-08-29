.PHONY: build clean install

build:
	@mkdir -p build && cd build && cmake .. && cmake --build .

clean:
	@rm -r build*/

install: build
	@cd build && cmake --install .
