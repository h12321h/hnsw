#include <iostream>
#include <fstream>
#include "src/hnsw.hpp"
#include "util/util.hpp"
#include "util/vecs_io.hpp"
#include "util/ground_truth.hpp"
#include "util/parameter.hpp"
#include<thread>
#include<mutex>

using namespace std;
using namespace HNSWLab;

HNSW hnsw;
vector <pair<vector<int>,int>> test_gnd_l_temp;
mutex mtx;

void process_query(const int *query, int dim,int i){
    vector<int> test_gnd = hnsw.query(query, dim);
    {
    lock_guard<mutex> lck (mtx);
    test_gnd_l_temp.push_back(make_pair(test_gnd,i));
    }
}

int main() {
    
    std::printf("load ground truth\n");
    int gnd_n_vec = 100;
    int gnd_vec_dim = 10;
    char *path = "./data/siftsmall/gnd.ivecs";
    int *gnd = read_ivecs(gnd_n_vec, gnd_vec_dim, path);

    std::printf("load query\n");
    int query_n_vec = 100;
    int query_vec_dim = 128;
    path = "./data/siftsmall/query.bvecs";
    int *query = read_bvecs(query_n_vec, query_vec_dim, path);

    std::printf("load base\n");
    int base_n_vec = 10000;
    int base_vec_dim = 128;
    path = "./data/siftsmall/base.bvecs";
    int *base = read_bvecs(base_n_vec, base_vec_dim, path);

    

    size_t report_every = 1000;
    TimeRecord insert_record;
    double single_insert_time;
    for (int i = 0; i < base_n_vec; i++) {
        hnsw.insert(base + base_vec_dim * i, i);

        if (i % report_every == 0) {
            single_insert_time += insert_record.get_elapsed_time_micro() / report_every;
            insert_record.reset();
        }
    }

    printf("average insert time %.1f μs\n", single_insert_time/(base_n_vec/report_every));


    printf("Parallel:querying\n");
    double single_query_time;
    vector<thread> threads;
    TimeRecord query_record;
    for (int i = 0; i < gnd_n_vec; i++) {
        threads.push_back(thread(process_query, query + i * query_vec_dim, gnd_vec_dim,i));
        //threads[i].join();
    }
    for (int i = 0; i < gnd_n_vec; i++) {
        threads[i].join();
    }
    single_query_time = query_record.get_elapsed_time_micro() / query_n_vec ;

    vector <vector<int>> test_gnd_l;
    //test_gnd_l_temp按照i排序
    sort(test_gnd_l_temp.begin(),test_gnd_l_temp.end(),[](pair<vector<int>,int> a,pair<vector<int>,int> b){return a.second<b.second;});
    for(auto i:test_gnd_l_temp){
        test_gnd_l.push_back(i.first);
    }
    double recall = count_recall(gnd_n_vec, gnd_vec_dim, test_gnd_l, gnd);
    printf("average recall: %.3f, single query time %.1f μs\n", recall, single_query_time);

    printf("Serial:querying\n");
    vector <vector<int>> test_gnd_lp;
    double single_query_timep;
    TimeRecord query_recordp;
    for (int i = 0; i < gnd_n_vec; i++) {
        vector<int> test_gnd = hnsw.query(query + i * query_vec_dim, gnd_vec_dim);
        test_gnd_lp.push_back(test_gnd);
    }
    single_query_timep = query_recordp.get_elapsed_time_micro() / query_n_vec;

    double recallp = count_recall(gnd_n_vec, gnd_vec_dim, test_gnd_lp, gnd);
    printf("average recall: %.3f, single query time %.1f μs\n", recallp, single_query_timep);

    return 0;
}