#pragma once


namespace matcha {

class Data {
  public:
    Data(void* data);

    void*  vs();

    float* fs();
    float* f32s();

    float  f();
    float  f32();

    int    i();
    int    i32();

    bool   b();
    bool   b8();

    operator float*();
    operator float();
    operator int();
    operator bool();

    float operator*();

  private:
    void* data_;
};

}
