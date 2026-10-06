#include "ttnte/sweep/sweep_graph.hpp"
#include "ttnte/utils/exception.hpp"
#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>

namespace ttnte::sweep {

SweepGraph::SweepGraph(int64_t num_nodes,
  std::vector<SweepInterface> interfaces, int64_t num_directions)
  : num_nodes_(num_nodes), num_directions_(num_directions),
    interfaces_(std::move(interfaces))
{
  const int64_t K = static_cast<int64_t>(interfaces_.size());
  std::vector<torch::Tensor> fw(K), bw(K);
  for (int64_t k = 0; k < K; k++) {
    const auto& itf = interfaces_[k];
    if (itf.forward_weight.size(0) != num_directions ||
        itf.backward_weight.size(0) != num_directions) {
      throw utils::runtime_error("ttnte::sweep::SweepGraph::SweepGraph",
        "Interface weights must have one entry per direction");
    }
    fw[k] = itf.forward_weight.to(torch::kFloat64).contiguous();
    bw[k] = itf.backward_weight.to(torch::kFloat64).contiguous();
  }

  std::map<std::vector<uint8_t>, size_t> signature_to_set;
  std::vector<std::vector<int64_t>> set_directions;
  direction_set_ = torch::empty({num_directions}, torch::kInt64);

  for (int64_t d = 0; d < num_directions; d++) {
    std::vector<EdgeState> fwd(K, EdgeState::NONE), bwd(K, EdgeState::NONE);

    // 1. Larger-weight rule on faces with flow both ways
    for (int64_t k = 0; k < K; k++) {
      const double tol = interfaces_[k].tolerance;
      const double a = fw[k].data_ptr<double>()[d];
      const double b = bw[k].data_ptr<double>()[d];
      const bool has_a = a > tol;
      const bool has_b = b > tol;
      if (has_a && has_b) {
        if (a >= b) {
          fwd[k] = EdgeState::DEPENDENCY;
          bwd[k] = EdgeState::LAGGED;
        } else {
          fwd[k] = EdgeState::LAGGED;
          bwd[k] = EdgeState::DEPENDENCY;
        }
        num_weight_lagged_++;
      } else {
        fwd[k] = has_a ? EdgeState::DEPENDENCY : EdgeState::NONE;
        bwd[k] = has_b ? EdgeState::DEPENDENCY : EdgeState::NONE;
      }
    }

    // 2. Lag DFS back edges of whatever cycles remain. Adjacency lists hold
    // (target, interface, is_forward).
    std::vector<std::vector<std::tuple<int64_t, int64_t, bool>>> adj(
      num_nodes_);
    for (int64_t k = 0; k < K; k++) {
      const auto& itf = interfaces_[k];
      if (fwd[k] == EdgeState::DEPENDENCY)
        adj[itf.first].emplace_back(itf.second, k, true);
      if (bwd[k] == EdgeState::DEPENDENCY)
        adj[itf.second].emplace_back(itf.first, k, false);
    }
    // 0 = unvisited, 1 = on stack, 2 = finished
    std::vector<uint8_t> color(num_nodes_, 0);
    for (int64_t root = 0; root < num_nodes_; root++) {
      if (color[root] != 0)
        continue;
      std::vector<std::pair<int64_t, size_t>> stack {{root, 0}};
      color[root] = 1;
      while (!stack.empty()) {
        auto& [node, next] = stack.back();
        if (next < adj[node].size()) {
          auto [target, k, is_forward] = adj[node][next++];
          if (color[target] == 1) {
            (is_forward ? fwd : bwd)[k] = EdgeState::LAGGED;
            num_dfs_lagged_++;
          } else if (color[target] == 0) {
            color[target] = 1;
            stack.emplace_back(target, 0);
          }
        } else {
          color[node] = 2;
          stack.pop_back();
        }
      }
    }

    // Group by signature
    std::vector<uint8_t> signature(2 * K);
    for (int64_t k = 0; k < K; k++) {
      signature[2 * k] = static_cast<uint8_t>(fwd[k]);
      signature[2 * k + 1] = static_cast<uint8_t>(bwd[k]);
    }
    auto [it, inserted] =
      signature_to_set.emplace(signature, angle_sets_.size());
    if (inserted) {
      AngleSet set;
      set.forward = fwd;
      set.backward = bwd;
      angle_sets_.push_back(std::move(set));
      set_directions.emplace_back();
    }
    set_directions[it->second].push_back(d);
    direction_set_[d] = static_cast<int64_t>(it->second);
  }

  for (size_t s = 0; s < angle_sets_.size(); s++) {
    angle_sets_[s].directions = torch::tensor(set_directions[s], torch::kInt64);
  }
}

std::vector<int64_t> SweepGraph::upstream(size_t angle_set, int64_t node) const
{
  const auto& set = angle_sets_.at(angle_set);
  std::vector<int64_t> nodes;
  for (size_t k = 0; k < interfaces_.size(); k++) {
    const auto& itf = interfaces_[k];
    if (itf.second == node && set.forward[k] == EdgeState::DEPENDENCY)
      nodes.push_back(itf.first);
    if (itf.first == node && set.backward[k] == EdgeState::DEPENDENCY)
      nodes.push_back(itf.second);
  }
  return nodes;
}

std::vector<int64_t> SweepGraph::topological_order(size_t angle_set) const
{
  std::vector<std::vector<int64_t>> downstream(num_nodes_);
  std::vector<int64_t> indegree(num_nodes_, 0);
  for (int64_t i = 0; i < num_nodes_; i++) {
    for (int64_t j : upstream(angle_set, i)) {
      downstream[j].push_back(i);
      indegree[i]++;
    }
  }
  std::vector<int64_t> order;
  order.reserve(num_nodes_);
  for (int64_t i = 0; i < num_nodes_; i++)
    if (indegree[i] == 0)
      order.push_back(i);
  for (size_t k = 0; k < order.size(); k++) {
    for (int64_t i : downstream[order[k]])
      if (--indegree[i] == 0)
        order.push_back(i);
  }
  if (static_cast<int64_t>(order.size()) != num_nodes_) {
    throw utils::runtime_error("ttnte::sweep::SweepGraph::topological_order",
      "The dependency graph of an angle set is cyclic");
  }
  return order;
}

std::vector<SweepChunk> SweepGraph::schedule(
  size_t angle_set, const std::vector<int64_t>& chunk_of) const
{
  if (static_cast<int64_t>(chunk_of.size()) != num_nodes_) {
    throw utils::runtime_error("ttnte::sweep::SweepGraph::schedule",
      "chunk_of must have one entry per node");
  }

  // Dense chunk ids and the chunk graph
  std::unordered_map<int64_t, int64_t> dense;
  std::vector<int64_t> chunk(num_nodes_);
  for (int64_t i = 0; i < num_nodes_; i++) {
    auto it = dense.emplace(chunk_of[i], static_cast<int64_t>(dense.size()));
    chunk[i] = it.first->second;
  }
  const int64_t C = static_cast<int64_t>(dense.size());
  std::vector<std::set<int64_t>> chunk_up(C);
  for (int64_t i = 0; i < num_nodes_; i++) {
    for (int64_t j : upstream(angle_set, i))
      if (chunk[j] != chunk[i])
        chunk_up[chunk[i]].insert(chunk[j]);
  }

  // Tarjan's strongly connected components over upstream edges (iterative).
  // Components are completed upstream-first, i.e. in topological order.
  std::vector<int64_t> comp(C, -1), low(C, 0), num(C, -1);
  std::vector<bool> on_stack(C, false);
  std::vector<int64_t> stack;
  int64_t counter = 0, num_comps = 0;
  std::vector<std::vector<int64_t>> up_list(C);
  for (int64_t c = 0; c < C; c++)
    up_list[c].assign(chunk_up[c].begin(), chunk_up[c].end());
  for (int64_t root = 0; root < C; root++) {
    if (num[root] >= 0)
      continue;
    std::vector<std::pair<int64_t, size_t>> call {{root, 0}};
    num[root] = low[root] = counter++;
    stack.push_back(root);
    on_stack[root] = true;
    while (!call.empty()) {
      auto& [v, next] = call.back();
      if (next < up_list[v].size()) {
        int64_t w = up_list[v][next++];
        if (num[w] < 0) {
          num[w] = low[w] = counter++;
          stack.push_back(w);
          on_stack[w] = true;
          call.emplace_back(w, 0);
        } else if (on_stack[w]) {
          low[v] = std::min(low[v], num[w]);
        }
      } else {
        const int64_t vv = v;
        if (low[vv] == num[vv]) {
          int64_t w;
          do {
            w = stack.back();
            stack.pop_back();
            on_stack[w] = false;
            comp[w] = num_comps;
          } while (w != vv);
          num_comps++;
        }
        call.pop_back();
        if (!call.empty()) {
          const int64_t parent = call.back().first;
          low[parent] = std::min(low[parent], low[vv]);
        }
      }
    }
  }

  // One task per component, units in global topological order
  std::vector<SweepChunk> tasks(num_comps);
  for (int64_t i : topological_order(angle_set))
    tasks[comp[chunk[i]]].units.push_back(i);
  for (int64_t c = 0; c < C; c++) {
    for (int64_t u : chunk_up[c]) {
      if (comp[u] != comp[c])
        tasks[comp[c]].upstream.push_back(comp[u]);
    }
  }
  for (auto& t : tasks) {
    std::sort(t.upstream.begin(), t.upstream.end());
    t.upstream.erase(
      std::unique(t.upstream.begin(), t.upstream.end()), t.upstream.end());
  }
  return tasks;
}

} // namespace ttnte::sweep
