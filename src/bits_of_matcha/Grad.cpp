#include "bits_of_matcha/Grad.h"



namespace matcha {

Grad::Grad(Flow* flow)
  : flow_{flow}
{}

void Grad::add(tensor* wrt) {

}

void Grad::add(std::vector<tensor*> wrt) {

}

std::vector<std::tuple<tensor*, tensor>> Grad::operator()() {
  return {};
}

}