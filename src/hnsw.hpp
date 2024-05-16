#pragma once
#include <iostream>
#include "base.hpp"
#include <vector>
#include <cstring>

#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <random>
#include <cassert>
#include <vector>
#include "../util/util.hpp"
#include "../util/vecs_io.hpp"
#include "../util/ground_truth.hpp"
#include "../util/parameter.hpp"

namespace HNSWLab
{

    struct node
    {
        const int *data;
        int id;
        int level;
        std::vector<std::vector<int>> neighbors; // 表示第lc层中节点q的邻居
        node(){};
    };

    class HNSW : public AlgorithmInterface
    {
    private:
        std::vector<node*> nodes;
        int entry_point = -1;
        const int dim = 128;

    public:
        // you can add more parameter to initialize HNSW
        HNSW(){};

        void insert(const int *item, int label);

        std::vector<int> query(const int *query, int k);

        std::vector<int> search_layer(const int *item, int ep, int ef, int level);
        std::vector<int> select_neighbors(const int *item, std::vector<int> &w, int ef, int level);

        ~HNSW(){};
    };

    /**
     * input:
     * iterm: the vector to be inserted
     * label: the label(id) of the vector
     */
    void HNSW::insert(const int *item, int label)
    {
        //  std::cout<<"item"<<item<<" "<<label<<std::endl;
        std::vector<int> w;
        int ep = this->entry_point;

        node* n=new node();
        n->data = item;
        n->id = label;

        if (ep == -1)
        {
            n->level = 0;
            n->neighbors.resize(1);
            nodes.push_back(n);
            entry_point = label;
            return;
        }

        int max_level = nodes[ep]->level;
        int l = get_random_level();

        n->level = l;
        n->neighbors.resize(l + 1);
        nodes.push_back(n);
        int i = max_level;
        for (; i > l; i--)
        {
            w = search_layer(item, ep, 1, i);
            ep = w[0];
        }

        for (; i >= 0; i--)
        {
            // std::cout<<"ep level"<<nodes[ep].level;
            // std::cout<<"  i level"<<i<<std::endl;

            w = search_layer(item, ep, ef_construction, i);

            n->neighbors[i] = select_neighbors(item, w, M, i); // 返回的是有序的
            w= n->neighbors[i];
            ep = n->neighbors[i][0];

            for (int j = 0; j < w.size(); j++)
            {
                nodes[w[j]]->neighbors[i].push_back(label);
                //  std::cout<<"w[j] "<<w[j]<<std::endl;
                if (nodes[w[j]]->neighbors[i].size() > M_max)
                {
                    // std::cout<<nodes[w[j]].neighbors[i].size()<<" "<<M_max<<std::endl;
                    //删除nodes[w[j]]->neighbors[i][M_max+1]这个点的邻居中的label
                    std::vector<int> temp=select_neighbors(nodes[w[j]]->data, nodes[w[j]]->neighbors[i], M_max, i);
                    int deleteNeighbor = nodes[w[j]]->neighbors[i][M_max];
                    for (int k = 0; k < nodes[deleteNeighbor]->neighbors[i].size(); k++)
                    {
                        if (nodes[deleteNeighbor]->neighbors[i][k] == label)
                        {
                            nodes[deleteNeighbor]->neighbors[i].erase(nodes[deleteNeighbor]->neighbors[i].begin() + k);
                            break;
                        }
                    }
                    nodes[w[j]]->neighbors[i] = temp;
                }
            }
        }

        if (l > max_level)
        {
            max_level = l;
            entry_point = label;
        }

        // std::cout<<"push:"<<label<<" "<<nodes[label].data<<std::endl;
    }

    /**
     * input:
     * query: the vector to be queried
     * k: the number of nearest neighbors to be returned
     *
     * output:
     * a vector of labels of the k nearest neighbors
     */
    std::vector<int> HNSW::query(const int *query, int k)
    {
        std::vector<int> res;
        std::vector<int> w;
        int ep = this->entry_point;
        int max_level = nodes[ep]->level;
        for (int i = max_level; i > 0; i--)
        {
            w = search_layer(query, ep, 1, i);
            ep = w[0];
        }

        w = search_layer(query, ep, ef_construction, 0);
        res = select_neighbors(query, w, k, 0);
        return res;
    }

    std::vector<int> HNSW::search_layer(const int *item, int ep, int ef, int level)
    {
        auto comparew = [](const std::pair<long, int> &a, const std::pair<long, int> &b)
        {
            return a.first < b.first;
        };

        auto comparec = [](const std::pair<long, int> &a, const std::pair<long, int> &b)
        {
            return a.first > b.first;
        };

        std::priority_queue<std::pair<long, int>, std::vector<std::pair<long, int>>, decltype(comparew)> w(comparew); // 大顶堆
        std::priority_queue<std::pair<long, int>, std::vector<std::pair<long, int>>, decltype(comparec)> c(comparec); // 小顶堆
        std::unordered_set<int> visited;

        visited.insert(ep);
        c.push(std::make_pair(l2distance(item, nodes[ep]->data, dim), ep));
        w.push(std::make_pair(l2distance(item, nodes[ep]->data, dim), ep));

        while (!c.empty())
        {
            auto q = c.top();
            c.pop();
            auto f = w.top();
            if (q.first > f.first)
            {
                break;
            }
            // if(nodes[q.second].neighbors.size() <=level){
            //     std::cout<<"error"<<std::endl;
            // }
            for (int i = 0; i < nodes[q.second]->neighbors[level].size(); i++)
            {
                int id = nodes[q.second]->neighbors[level][i];
                if (visited.find(id) == visited.end())
                {
                    visited.insert(id);
                    long dis = l2distance(item, nodes[id]->data, dim);
                    if (w.size() < ef)
                    {
                        w.push(std::make_pair(dis, id));
                    }
                    else if (dis < w.top().first)
                    {
                        w.pop();
                        w.push(std::make_pair(dis, id));
                    }
                    c.push(std::make_pair(dis, id));
                }
            }
        }
        std::vector<int> res;
        // std::cout<<w.size()<<" "<<ef<<std::endl;
        while (!w.empty() && res.size() < ef)
        {
            // std::cout<<w.top().first<<std::endl;
            res.push_back(w.top().second);
            w.pop();
        }
        //  std::cout<<std::endl;
        return res;
    }

    std::vector<int> HNSW::select_neighbors(const int *item, std::vector<int> &w, int ef, int level)
    {
        auto compare = [](const std::pair<long, int> &a, const std::pair<long, int> &b)
        {
            return a.first > b.first;
        };
        std::priority_queue<std::pair<long, int>, std::vector<std::pair<long, int>>, decltype(compare)> q(compare);

        for (int i = 0; i < w.size(); i++)
        {
            // std::cout<<i<<" "<<w[i]<<" "<<nodes[w[i]].data<<std::endl;
            q.push(std::make_pair(l2distance(item, nodes[w[i]]->data, dim), w[i]));
        }
        std::vector<int> res;
        w.clear();
        while (!q.empty() && res.size() < ef)
        {
            //  std::cout<<q.top().first<<std::endl;
            res.push_back(q.top().second);
            w.push_back(q.top().second);
            q.pop();
        }
        while(!q.empty()){
            w.push_back(q.top().second);
            q.pop();
        }
        // std::cout<<std::endl;

        return res;
    }

}
