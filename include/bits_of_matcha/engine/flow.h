#pragma once


namespace matcha::engine {

class Flow {
  public:
    Flow();

    bool built();
    void check();

  private:
    bool built_;

};

}