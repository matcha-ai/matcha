#include "bits_of_matcha/nn/callbacks/Logger.h"
#include "bits_of_matcha/nn/Net.h"
#include "bits_of_matcha/print.h"

#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>
#include <iomanip>
#include <mutex>


namespace matcha::nn {

struct Internal : Callback {
  Internal(std::ostream& stream)
    : line(stream)
    , updaterIntervalMs(30)
    , spinner(updaterIntervalMs)
    , eta_(1000)
  {
  }

  class Line {
  public:
    Line(std::ostream& stream)
      : os_(stream)
      , lastSize_(0)
    {}

    template <class T>
    Line& operator<<(const T& t) {
      ss_ << t;
      return *this;
    }

    void flush() {
      std::string msg = ss_.str();
//      os_ << std::string(lastSize_, '\b');
      os_ << "\r";
      os_ << msg << std::flush;
      if (lastSize_ > msg.size()) {
        size_t toErase = lastSize_ - msg.size();
        os_ << std::string(toErase, ' ');
        os_ << std::string(toErase, '\b');
      }
      lastSize_ = msg.size();
      ss_.str("");
    }

    void endl() {
      os_ << ss_.str() << std::endl;
      ss_.str("");
      lastSize_ = 0;
    }

  private:
    std::stringstream ss_;
    std::ostream& os_;
    size_t lastSize_;

  };

  template <class T>
  class Interval {
  public:
    Interval(size_t ms) : interval_(ms) {};

    bool action() {
      auto now = std::chrono::steady_clock::now();
      auto duration = now - time_;
      size_t passed = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
      if (passed >= interval_) {
        time_ = now;
        return true;
      }
      return false;
    }

    bool operator==(const T& t) const {
      return value_ == t;
    }

    bool operator!=(const T& t) const {
      return value_ != t;
    }

    Interval& operator=(const T& t) {
      value_ = t;
      return *this;
    }

    operator T() {
      return value_;
    }

  private:
    T value_;
    std::chrono::time_point<std::chrono::steady_clock> time_;
    size_t interval_;
  };

  class Spinner {
  private:
    static constexpr const char* seq[] = {
      "⡿",
      "⣟",
      "⣯",
      "⣷",
      "⣾",
      "⣽",
      "⣻",
      "⢿"
    };
    static constexpr const size_t seqCount = sizeof(seq) / sizeof(seq[0]);

  public:
    Spinner(size_t intervalMs) {
      pos_ = 0;
      intervalMs_ = intervalMs;
    }

    const char* render() {
      auto now = std::chrono::steady_clock::now();
      auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_).count();
      if (millis > intervalMs_) {
        pos_ = (pos_ + 1) % seqCount;
        last_ = now;
      }
      return seq[pos_];
    }

    size_t pos_;
    size_t intervalMs_;
    std::chrono::time_point<std::chrono::steady_clock> last_;
  };

  void onfitInit(Net& net) override {
    std::cout << "fitting matcha::Net ";
    std::cout << std::flush;
  }

  void onfitBegin(Net& net, Dataset ds) override {
    std::cout << "(" << net.params.total() << " trainable parameters) ";
    std::cout << std::endl;
  }

  void onEpochBegin(size_t epoch, size_t max) override {
    epochBeginTime_ = std::chrono::steady_clock::now();
    epoch_ = epoch + 1;
    epochs_ = max;
    size_t intervalMs = updaterIntervalMs;
    eta_ = "";
    eta_.action();
    batches_ = -1;
    updater_ = (std::thread) [&, epoch_ = epoch_, intervalMs]() {
      while (this->epoch_ == epoch_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(intervalMs));
        if (batches_ == -1) continue;
        this->update();
      }
    };
  }

  void onEpochEnd() override {
    updater_.detach();
    size_t epadding = std::to_string(epochs_).size();
    auto estring = std::to_string(epoch_);
    line << " *  ";
    line << "epoch " << std::string(epadding - estring.size(), ' ') << estring;
    line << "  ::  finished";
    line.flush();
    line.endl();
  }

  void onBatchBegin(size_t batch, size_t max) override {
    batch_ = batch;
    batches_ = max;
    batchBeginTime_ = std::chrono::steady_clock::now();
    update();
  }

  void onBatchEnd() override {
    auto eduration = std::chrono::steady_clock::now() - epochBeginTime_;
    auto bduration = std::chrono::steady_clock::now() - batchBeginTime_;
    usEpoch_ = std::chrono::duration_cast<std::chrono::microseconds>(eduration).count();
    usBatch_ = std::chrono::duration_cast<std::chrono::microseconds>(bduration).count();
  }

  void onPropagateForward(const Instance& instance, const tensor& loss) override {
    if (!epoch_) return;
    std::stringstream buff;
    loss_ = loss;
    buff << loss;
    float len = (float) buff.str().size();

//    float coef = std::min(100.0 / batches_, .05);
    float coef = .1;
    if (batch_ < 10)
      lossLen_ -= (float) 1. / (float) (batch_ + 1) * (lossLen_ - len);
    else
      lossLen_ = lossLen_ * (1 - coef) + len * coef;
  }

  void onPropagateBackward(const std::map<tensor*, tensor>& grads) override {
    if (!epoch_) return;
    if (grads.empty()) return;
    std::vector<tensor> gflows;
    for (auto&& [t, g]: grads) gflows.push_back(l2norm(g) / g.size());
    gflowM_ = mean(stack(gflows));
    gflowSd_ = stdevu(stack(gflows));
  }

  void update() {
    std::lock_guard guard(mtx_);
    size_t bpadding = std::to_string(batches_).size();
    size_t epadding = std::to_string(epochs_).size();
    auto bstring = std::to_string(batch_ + 1);
    auto estring = std::to_string(epoch_);
    line << " " << spinner.render() << "  ";
    line << "epoch " << std::string(epadding - estring.size(), ' ') << estring
         << "/" << epochs_ << " ";
    line << "  batch " << std::string(bpadding - bstring.size(), ' ') << bstring
         << "/" << batches_ << " ";

    if (batch_ != 0 && eta_.action()) {
      std::stringstream  ss_;
//      ss_ << std::fixed << std::setprecision(2);
      ss_ << " ::  ";
//      << ((float) usBatch_ / 1000.0)  << " ms, ";
      if (batch_ == lastEtaBatch_) {
        if (sEta_ >= 3) sEta_ = sEta_ - 1;
      } else {
        lastEtaBatch_ = batch_;
        sEta_ = (float) (batches_ - batch_) / batch_ * (usEpoch_ / 1000) / 1000.;
      }

      float etas = sEta_;

      ss_ << "ETA ";
      int h = (int) etas / 3600;
      etas -= (float) h * 3600;
      int min = (int) etas / 60;
      etas -= (float) min * 60;
      int s = (int) etas;
      if (h) ss_ << h << " h ";
      if (h || min) ss_ << min << " min ";
      ss_ << s << " s ";
      eta_ = ss_.str();
    }

    line << (std::string) eta_;

    if (batch_ != 0) {
      std::stringstream buff;
      buff << loss_;
      line << " ::  loss "
           << buff.str();
      size_t padding = (size_t) lossLen_ + 2;
      if (padding > buff.str().length())
        line << std::string(padding - buff.str().length(), ' ');
      line <<  " ";
      line << " ::  grads " << gflowSd_ << " +- " << gflowSd_ << " ";
    }

    line.flush();
  }

  size_t updaterIntervalMs;

  Line line;
  Spinner spinner;

  size_t batch_ = 0;
  size_t batches_ = 0;
  size_t epoch_ = 0;
  size_t epochs_ = 0;
  size_t usEpoch_ =0;
  size_t usBatch_ = 0;
  size_t lastEtaBatch_ =0;
  float sEta_ = 0;
  float lossLen_ = 0;
  tensor loss_;
  tensor gflowM_;
  tensor gflowSd_;
  Net* net_;

  Interval<std::string> eta_;

  std::chrono::time_point<std::chrono::steady_clock> epochBeginTime_;
  std::chrono::time_point<std::chrono::steady_clock> batchBeginTime_;

  std::thread updater_;
  std::mutex mtx_;
};

Logger::operator std::shared_ptr<Callback>() {
  auto internal = new Internal(stream);
  return std::shared_ptr<Callback>(internal);
}

}