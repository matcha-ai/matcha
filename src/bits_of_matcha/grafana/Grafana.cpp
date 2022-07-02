#include "bits_of_matcha/grafana/Grafana.h"
#include "bits_of_matcha/print.h"
//#include <dlfcn.h>
#include <curl/curl.h>
#include <sstream>

using namespace matcha;

Grafana::Grafana(const std::string& url, const std::string& oauth)
{
  curl_global_init(CURL_GLOBAL_ALL);

  url_ = url;
  oauth_ = oauth;

  if (url_.back() == '/') url_.erase(url.end() - 1);

  auto r = httpGet(url_ + "/api/datasources", oauth_);
  print("|" + r + "|");
}

Datasource Grafana::create(const std::string& name) {
  std::stringstream ss;
  ss << "{ " << R"("name": ")" << name << "\", "
     << R"("type": "grafana" })";
  auto res = httpPost(url_ + "/api/datasources", oauth_, ss.str());
  print(res);
  return {};
}

size_t writefunc(void *ptr, size_t size, size_t nmemb, std::string& s)
{
  s = (char*) ptr;
  return size*nmemb;
}

std::string Grafana::httpPost(const std::string& url, const std::string& oauth, const std::string& data) {
  CURL* curl = curl_easy_init();
  if (!curl) throw std::runtime_error("curl error");

  const std::string& rbuff = data;
  std::string wbuff;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_XOAUTH2_BEARER, oauth.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &wbuff);
//  curl_easy_setopt(curl, CURLOPT_READFUNCTION
  curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BEARER);
  curl_slist* header = nullptr;
  header = curl_slist_append(header, "Content-Type: application/json");
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, rbuff.c_str());

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    throw std::runtime_error("curl request error");
  }
  curl_easy_cleanup(curl);
  return wbuff;
}

std::string Grafana::httpGet(const std::string& url, const std::string& oauth) {
  CURL* curl = curl_easy_init();
  if (!curl) throw std::runtime_error("curl error");

  std::string wbuff;

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_XOAUTH2_BEARER, oauth.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &wbuff);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writefunc);
  curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BEARER);
  curl_slist* header = nullptr;
  header = curl_slist_append(header, "Content-Type: application/json");
  header = curl_slist_append(header, "Accept: application/json");
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header);

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    throw std::runtime_error("curl request error");
  }
  curl_easy_cleanup(curl);
  return wbuff;
}