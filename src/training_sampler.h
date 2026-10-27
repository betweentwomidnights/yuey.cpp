#pragma once
#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace yue2::detail {
class TrainingSampler {
  public:
    TrainingSampler(size_t count, bool shuffle, uint32_t seed,
                    std::vector<size_t> groups = {})
        : count_(count), shuffle_(shuffle), rng_(seed), groups_(std::move(groups)) {
        if (!count_)
            throw std::invalid_argument("empty sampler");
        if(!groups_.empty()) {
            if(groups_.size()!=count_||!shuffle_)throw std::invalid_argument("grouped sampling requires shuffle and one group per sample");
            const auto count=*std::max_element(groups_.begin(),groups_.end())+1;
            if(count>count_)throw std::invalid_argument("invalid sample groups");
            members_.resize(count);member_order_.resize(count);member_cursor_.resize(count,0);
            for(size_t i=0;i<count_;++i)members_[groups_[i]].push_back(i);
            for(const auto &group:members_)if(group.empty())throw std::invalid_argument("sample groups must be contiguous");
        }
    }
    size_t next() {
        if(!groups_.empty()) {
            if(cursor_==order_.size()) {
                order_.resize(members_.size());std::iota(order_.begin(),order_.end(),size_t(0));
                std::shuffle(order_.begin(),order_.end(),rng_);cursor_=0;
            }
            const auto group=order_[cursor_++];
            if(member_cursor_[group]==member_order_[group].size()) {
                member_order_[group]=members_[group];std::shuffle(member_order_[group].begin(),member_order_[group].end(),rng_);member_cursor_[group]=0;
            }
            return member_order_[group][member_cursor_[group]++];
        }
        if (!shuffle_)
            return count_ == 1 ? 0 : std::uniform_int_distribution<size_t>(0, count_ - 1)(rng_);
        if (cursor_ == order_.size()) {
            order_.resize(count_);
            std::iota(order_.begin(), order_.end(), size_t(0));
            std::shuffle(order_.begin(), order_.end(), rng_);
            cursor_ = 0;
        }
        return order_[cursor_++];
    }
    std::string save() const {
        std::ostringstream s;
        if(!groups_.empty()) {
            s<<"ar-grouped-sampler-v1 "<<count_<<' '<<members_.size()<<' '<<cursor_<<' '<<order_.size()<<'\n';
            for(auto group:groups_)s<<group<<' ';s<<'\n';
            for(auto group:order_)s<<group<<' ';s<<'\n';
            for(size_t group=0;group<members_.size();++group) {
                s<<member_cursor_[group]<<' '<<member_order_[group].size()<<' ';
                for(auto sample:member_order_[group])s<<sample<<' ';s<<'\n';
            }
            s<<rng_;return s.str();
        }
        s << "ar-sampler-v3 " << count_ << ' ' << shuffle_ << ' ' << cursor_ << ' ' << order_.size()
          << '\n';
        for (auto i : order_)
            s << i << ' ';
        s << '\n' << rng_;
        return s.str();
    }
    void load(const std::string &state) {
        if(!groups_.empty()) {
            std::istringstream input(state);std::string marker;size_t count,group_count,cursor,n;
            if(!(input>>marker>>count>>group_count>>cursor>>n)||marker!="ar-grouped-sampler-v1"||count!=count_||group_count!=members_.size()||cursor>n||(n&&n!=group_count))throw std::runtime_error("incompatible grouped sampler state");
            for(auto expected:groups_){size_t actual;if(!(input>>actual)||actual!=expected)throw std::runtime_error("sample group assignment changed");}
            std::vector<size_t> order(n);std::vector<bool> seen(group_count,false);
            for(auto &group:order){if(!(input>>group)||group>=group_count||seen[group])throw std::runtime_error("invalid group permutation");seen[group]=true;}
            auto member_order=member_order_;auto member_cursor=member_cursor_;
            for(size_t group=0;group<group_count;++group) {
                size_t m;if(!(input>>member_cursor[group]>>m)||member_cursor[group]>m||(m&&m!=members_[group].size()))throw std::runtime_error("invalid group cursor");
                member_order[group].resize(m);std::vector<bool> used(count_,false);
                for(auto &sample:member_order[group]){if(!(input>>sample)||sample>=count_||groups_[sample]!=group||used[sample])throw std::runtime_error("invalid within-group permutation");used[sample]=true;}
            }
            auto rng=rng_;if(!(input>>rng))throw std::runtime_error("invalid grouped RNG");
            rng_=rng;order_=std::move(order);cursor_=cursor;member_order_=std::move(member_order);member_cursor_=std::move(member_cursor);return;
        }
        auto rng = rng_;
        std::vector<size_t> order;
        size_t cursor = 0;
        if (state == "single-sample-v1" && count_ == 1 && !shuffle_)
            return;
        if (state.rfind("ar-sampler-v2\n", 0) == 0 && !shuffle_) {
            std::istringstream s(state.substr(std::string("ar-sampler-v2\n").size()));
            if (!(s >> rng))
                throw std::runtime_error("invalid legacy RNG");
        } else {
            std::istringstream s(state);
            std::string marker;
            size_t count, n;
            bool shuffle;
            if (!(s >> marker >> count >> shuffle >> cursor >> n) || marker != "ar-sampler-v3" ||
                count != count_ || shuffle != shuffle_ || n > count_ || cursor > n ||
                (n && n != count_) || (!shuffle_ && (n || cursor)))
                throw std::runtime_error("incompatible sampler state");
            order.resize(n);
            std::vector<bool> seen(count_, false);
            for (auto &i : order) {
                if (!(s >> i) || i >= count_ || seen[i])
                    throw std::runtime_error("invalid sampler permutation");
                seen[i] = true;
            }
            if (!(s >> rng))
                throw std::runtime_error("invalid sampler RNG");
        }
        rng_ = rng;
        order_ = std::move(order);
        cursor_ = cursor;
    }

  private:
    size_t count_, cursor_ = 0;
    bool shuffle_;
    std::mt19937 rng_;
    std::vector<size_t> order_;
    std::vector<size_t> groups_;
    std::vector<std::vector<size_t>> members_,member_order_;
    std::vector<size_t> member_cursor_;
};
} // namespace yue2::detail
