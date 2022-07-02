#pragma once

#include <string>
#include "bits_of_matcha/tensor.h"


class Datasource {
public:

private:
  std::string url_, oauth_;
  int id;
};

class Grafana {
public:
  Grafana(const std::string& url, const std::string& oauth);

  bool exists(const std::string& name);
  Datasource get(const std::string& name);
  Datasource create(const std::string& name);

private:
  std::string url_, oauth_;
  static std::string httpPost(const std::string& url, const std::string& oauth, const std::string& data);
  static std::string httpGet(const std::string& url, const std::string& oauth);
  friend class Datasource;
};
