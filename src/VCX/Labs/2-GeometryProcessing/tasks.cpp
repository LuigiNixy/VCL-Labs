#include <iostream>
#include <list>
#include <map>
#include <set>
#include <unordered_set>
<<<<<<< HEAD
#include <algorithm>
#include <queue>
#include <glm/gtx/string_cast.hpp>
=======
>>>>>>> c1db74a9bd05d4381ce103d6e5f5baffd5222371

#include <glm/gtc/matrix_inverse.hpp>
#include <spdlog/spdlog.h>

#include "Labs/2-GeometryProcessing/DCEL.hpp"
#include "Labs/2-GeometryProcessing/tasks.h"

<<<<<<< HEAD

namespace VCX::Labs::GeometryProcessing {

#include "Labs/2-GeometryProcessing/marching_cubes_table.h"
using namespace std;
=======
namespace VCX::Labs::GeometryProcessing {

#include "Labs/2-GeometryProcessing/marching_cubes_table.h"
>>>>>>> c1db74a9bd05d4381ce103d6e5f5baffd5222371

    /******************* 1. Mesh Subdivision *****************/
    void SubdivisionMesh(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, std::uint32_t numIterations) {
        // your code here
<<<<<<< HEAD

        
        map<pair<size_t,size_t>,size_t> newVertice;

        output.Indices = input.Indices;
        output.Positions = input.Positions;

        std::uint32_t curIteration = 0;
        while(curIteration <= numIterations){
            curIteration ++;
            DCEL links;
            links.AddFaces(output.Indices);
            if(! links.IsValid())
                 return;

            vector<glm::vec3> tmp_out;
            vector<uint32_t> tmp_ind;

            tmp_out = output.Positions;

            for(DCEL::HalfEdge const * e : links.GetEdges()){
                glm::vec3 newVertex(0);
                newVertex = glm::vec3(0.375f) * (output.Positions[e->To()] + output.Positions[e->From()]);
                if(e->PairEdge() != NULL){
                    newVertex += glm::vec3(0.125f) *
                        (output.Positions[e->OppositeVertex()] + output.Positions[e->PairOppositeVertex()]);
                }
                else newVertex += glm::vec3(0.125f) * (output.Positions[e->To()] + output.Positions[e->From()]);

                if(newVertice.find( make_pair(e->From(),e->To()) ) == newVertice.end() ){
                    tmp_out.push_back(newVertex);
                    newVertice[make_pair(e->To(),e->From())] = tmp_out.size() - 1;
                }                
            }

            for(std::size_t i = 0; i < output.Positions.size(); i++){
                DCEL::Vertex v = links.GetVertex(i);
                double u;
                int n = v.GetNeighbors().size();
                if( n == 3) u = 3.0f/16;
                else u = 3/((double)n * 8);
                glm::vec3 newPos(0,0,0);
                for(auto && a : v.GetNeighbors()){
                    newPos = newPos + glm::vec3(u) * output.Positions[a];    
                }
                tmp_out[i] = glm::vec3(1 - n*u) * output.Positions[i] + newPos;
            }

        size_t * p = new size_t[sizeof(size_t)*6];
        DCEL::HalfEdge const ** e = new DCEL::HalfEdge const *[sizeof(void *)*3];

        for(auto && f : links.GetFaces()){
            
            for(size_t i = 0; i <=2; i++){
                p[i] = *(f.Indices(i));
                e[i] = f.Edges(i);
            }

            for(size_t i = 0; i<=2; i++){
                if(newVertice.find(make_pair(e[i]->To(),e[i]->From())) != newVertice.end()){
                    p[i + 3] = newVertice[make_pair(e[i]->To(),e[i]->From())];
                }
                else p[i + 3] = newVertice[make_pair(e[i]->From(),e[i]->To())];
            }

            tmp_ind.push_back(p[0]);
            tmp_ind.push_back(p[5]);
            tmp_ind.push_back(p[4]);

            tmp_ind.push_back(p[5]);
            tmp_ind.push_back(p[3]);
            tmp_ind.push_back(p[4]);

            tmp_ind.push_back(p[5]);
            tmp_ind.push_back(p[1]);
            tmp_ind.push_back(p[3]);

            tmp_ind.push_back(p[4]);
            tmp_ind.push_back(p[3]);
            tmp_ind.push_back(p[2]);

        }
        
        delete []p;
        delete []e;

        output.Positions = tmp_out;
        output.Indices = tmp_ind;
        }

=======
>>>>>>> c1db74a9bd05d4381ce103d6e5f5baffd5222371
    }

    /******************* 2. Mesh Parameterization *****************/
    void Parameterization(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, const std::uint32_t numIterations) {
        // your code here
<<<<<<< HEAD
        DCEL links;
        
        links.AddFaces(input.Indices);
        if(!links.IsValid())
            return;

        output.Indices = input.Indices;
        output.Positions = input.Positions;
        output.TexCoords = input.TexCoords;

        std::size_t pointSize = input.Positions.size();

        std::size_t interiorSize = 0;
        vector<size_t> interiorPoint;
        float min_x,min_y,max_x,max_y;
        min_x = max_x = input.Positions[0].x;
        min_y = max_y = input.Positions[0].y;
        map<size_t,size_t> Idx;

        //find interior points
        for(std::size_t i = 0; i < pointSize; ++i){
            DCEL::Vertex v = links.GetVertex(i);
            output.TexCoords.push_back(glm::vec2(input.Positions[i].x, input.Positions[i].y));
            if(v.IsSide()) {
                continue;
            }
            interiorSize ++;
            interiorPoint.push_back(i);
            Idx[i] = interiorSize -1;
        }

        //Initialize boundary points 
        for(std::size_t i = 0; i < pointSize; ++i){
            DCEL::Vertex v = links.GetVertex(i);
            if(v.IsSide()){
                float tx = input.Positions[i].x;
                float ty = input.Positions[i].y;
                float c = tx/(sqrt(tx * tx + ty * ty));
                float s = ty/(sqrt(tx * tx + ty * ty));
                output.TexCoords[i] = glm::vec2(0.5f) + glm::vec2(0.5f) * glm::vec2(c,s);
            }

        }
            
        glm::vec2 * b = new glm::vec2[sizeof(glm::vec2)*interiorSize];
        glm::vec2 * g = new glm::vec2[sizeof(glm::vec2)*interiorSize];
        vector<std::size_t> pointIdx;



        // set b & g
        memset(b,0,sizeof(glm::vec2)*interiorSize);

        for(size_t i = 0; i < interiorSize; ++i){
            g[i] = glm::vec2(0);
            DCEL::Vertex v = links.GetVertex(interiorPoint[i]);
            for(auto && it : v.GetNeighbors()){
                    DCEL::Vertex u = links.GetVertex(it);
                    if(u.IsSide()){
                        b[i] = b[i] + glm::vec2(1.0f/v.GetNeighbors().size()) * output.TexCoords[it];
                    }
                }                
        }

        // Jacobi iteration, solve Ag = b
        for (int iter = 0; iter < numIterations; ++iter) {
            glm::vec2 * g1 = new glm::vec2[sizeof(glm::vec2)*interiorSize];
            memset(g1,0,sizeof(glm::vec2)*interiorSize);
            for(std::size_t x = 0; x < interiorSize; x++){
                glm::vec2 tmp(0,0);
                DCEL::Vertex v = links.GetVertex(interiorPoint[x]);
                for(auto && it : v.GetNeighbors()){
                    DCEL::Vertex u = links.GetVertex(it);
                    if(!u.IsSide()){
                        tmp += glm::vec2(1.0f/v.GetNeighbors().size(),1.0f/v.GetNeighbors().size()) * g[Idx[it]];
                    }
                }    
                g1[x] = tmp + b[x];
            }
            for(std::size_t x = 0; x < interiorSize; x++) g[x] = g1[x];
        }
        size_t i = 0;
        for (std::size_t x = 0; x < pointSize; ++x ){
            DCEL::Vertex v = links.GetVertex(x);
            if(!v.IsSide()) output.TexCoords[x] = g[i++];
        }

        delete[] g;
        delete[] b;
        }
    /******************* 3. Mesh Simplification *****************/
    
    void vec3out(glm::vec3 a){
        cout<<a.x<<" "<<a.y<<" "<<a.z<<endl;
        return;
    }
    void SimplifyMesh(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, float valid_pair_threshold, float simplification_ratio) {
        // your code here

        class validPair{
        public:
            float cost;
            size_t x,y;
            glm::vec3 v_bar;
            friend Engine::SurfaceMesh;
            validPair(float _cost = 0.0f, size_t _x = 0, size_t _y = 0, glm::vec3 _v_bar = glm::vec3(0)){
                cost = _cost;
                x = _x;
                y = _y;
                if(x > y) swap(x,y);
                v_bar = _v_bar;
            }
            // bool ty();
            bool operator <(const validPair & p) const{
                long long t1 = (x << 32) + y;
                long long t2 = (p.x << 32) + p.y;
                return t1 < t2;
            } 
        };

        // bool validPair::ty(){
        //     v1 = input.Positions[x];
        //     return *this;
        // }
        
        // std::cout<<simplification_ratio<<endl;
        int initialSize = input.Positions.size();

        output.Indices = input.Indices;
        output.Positions = input.Positions;

        DCEL links;
        links.AddFaces(output.Indices);
        if(!links.IsValid())
            return;

        vector<validPair> validPairs;
        vector<glm::mat4> q;
        size_t newIdx = -1;

        for(std::size_t i = 0; i < output.Positions.size(); i++){
                DCEL::Vertex v = links.GetVertex(i);
                glm::mat4 Qi(0.0f);
                for(auto && a : v.GetFaces()){
                    glm::vec3 p1 = output.Positions[*(a->Indices(0))];
                    glm::vec3 p2 = output.Positions[*(a->Indices(1))];
                    glm::vec3 p3 = output.Positions[*(a->Indices(2))];
                    glm::vec3 normal = glm::cross(p2-p1,p3-p1);
                    normal = glm::normalize(normal);
                    float d = - glm::dot(p1,normal);
                    glm::vec4 p = glm::vec4(normal.x,normal.y,normal.z,d);
                    Qi = Qi + glm::outerProduct(p, p);
                }
                q.push_back(Qi);
            }
        

        for(size_t x = 0; x <output.Positions.size(); x++){
            DCEL::Vertex v = links.GetVertex(x);
            for(auto && y : v.GetNeighbors()){
                glm::vec3 p1 = output.Positions[x];
                glm::vec3 p2 = output.Positions[y];

                glm::mat4 Q_bar = q[x] + q[y];
                // Q_bar[0][3] = 0;
                // Q_bar[1][3] = 0;
                // Q_bar[2][3] = 0;
                // Q_bar[3][3] = 1;

                glm::mat3 Q_inv = glm::inverse(glm::mat3(Q_bar));
                glm::vec3 v_bar = Q_inv * glm::vec3(Q_bar[3][0],Q_bar[3][1],Q_bar[3][2]);
                v_bar = v_bar * glm::vec3(-1.0f);
                // if(x <= y)
                    validPairs.push_back(validPair(glm::dot(glm::vec4(v_bar,1)* (q[x] + q[y]) ,glm::vec4(v_bar,1)), x, y, v_bar));
                // else 
                    // validPairs.push_back(validPair(glm::dot(v_bar* (q[x] + q[y]) ,v_bar), y, x, v_bar));
               }
        }

        for(size_t x = 0; x < output.Positions.size(); x++)
                for(size_t y = x + 1; y < output.Positions.size(); y++){
                    DCEL::Vertex v = links.GetVertex(x);
                    glm::vec3 p1 = output.Positions[x];
                    glm::vec3 p2 = output.Positions[y];

                    glm::mat4 Q_bar = q[x] + q[y];
                    Q_bar[0][3] = 0;
                    Q_bar[1][3] = 0;
                    Q_bar[2][3] = 0;
                    Q_bar[3][3] = 1;

                    // glm::vec4 v_bar = glm::vec4(0,0,0,1) * glm::inverse(Q_bar) ;
                     glm::mat3 Q_inv = glm::inverse(glm::mat3(Q_bar));
                glm::vec3 v_bar = Q_inv * glm::vec3(Q_bar[3][0],Q_bar[3][1],Q_bar[3][2]);

                    v_bar = v_bar * glm::vec3(-1.0f);
                    // cout<<glm::dot((v_bar * Q_bar) ,v_bar)<<endl;
                    // return;
                    if(glm::dot(output.Positions[x]-output.Positions[y],output.Positions[x]-output.Positions[y]) < valid_pair_threshold) {
                        // validPairs.push_back(validPair(glm::dot((v_bar * (q[x] + q[y])) ,v_bar), x, y, v_bar));
                        validPairs.push_back(validPair(glm::dot(glm::vec4(v_bar,1)* (q[x] + q[y]) ,glm::vec4(v_bar,1)), x, y, v_bar));
                        continue;
                    }
                }
        for(auto it = validPairs.begin(); it != validPairs.end();it ++){
                for(auto jt = it + 1; jt != validPairs.end();){
                    if(it->x == jt ->x && it->y == jt->y) {
                        validPairs.erase(jt);
                    }
                    else jt ++;
                }
            }
        
        // return;
        while( (float)output.Positions.size()/initialSize > simplification_ratio){
        // for(int seee = 0; seee <= 200; seee++){
           
            // std::cout<<output.Positions.size()<<endl;
            DCEL tmpLinks;
            tmpLinks.AddFaces(output.Indices);

            // if(!tmpLinks.IsValid()) {
            //     cout<<"error"<<endl;
            //     return;
            // }

            //find the pair with the least error
            float minCost = validPairs[0].cost;
            auto argmin = validPairs[0];
            for(auto && i : validPairs){
                // cout<<i.cost<<endl;
                if(i.cost < minCost){
                    minCost = i.cost;
                    argmin = i;
                    // vec3out(argmin.v_bar);
                }
            }

            //update Indices
            for(size_t i = 0; i < output.Indices.size();){
                int flag1 = -1, flag2 = -1;
                for(int l = 0; l <=2 ; l++){
                    if(output.Indices[i+l] == argmin.x) flag1 = l;
                    if(output.Indices[i+l] == argmin.y) flag2 = l;
                }
                if(flag1 != -1 && flag2 != -1) {
                    output.Indices.erase(i + output.Indices.begin(), i + 3 + output.Indices.begin());
                    continue;
                }
                if(flag1 != -1){
                    output.Indices[i + flag1] = output.Positions.size();
                }
                else if(flag2 != -1){
                    output.Indices[i + flag2] = output.Positions.size();
                }
                for(int l = 0; l <=2 ; l++){
                    if(output.Indices[i+l] >= argmin.x) output.Indices[i + l] --;
                    if(output.Indices[i+l] >= argmin.y) output.Indices[i + l] --;
                }
                i += 3;
            }

            for(size_t i = 0; i <output.Indices.size(); i++){
                if(output.Indices[i] >= output.Positions.size() - 2){
                    // cout<<i<<endl;
                    output.Indices[i] = output.Positions.size() - 2;

                }
                    
            }
            
            q.push_back(q[argmin.x] + q[argmin.y]);
            output.Positions.push_back(argmin.v_bar);


            size_t tmp = output.Positions.size() ;

            //update valid pairs

            for(auto it = validPairs.begin(); it != validPairs.end();){
                if(it->x == argmin.x && it->y == argmin.y){
                    it = validPairs.erase(it);
                    continue;
                }
                if(it->x == argmin.x || it->x == argmin.y){
                    it->x = it->y;
                    it->y = tmp;

                    glm::mat4 Q_bar = q[it->x] + q[it->y - 1];
                    Q_bar[0][3] = 0;
                    Q_bar[1][3] = 0;
                    Q_bar[2][3] = 0;
                    Q_bar[3][3] = 1;

                    glm::mat3 Q_inv = glm::inverse(glm::mat3(Q_bar));
                    glm::vec3 v_bar = Q_inv * glm::vec3(Q_bar[3][0],Q_bar[3][1],Q_bar[3][2]);
                    
                    v_bar = v_bar * glm::vec3(-1.0f);
                    it->cost = glm::dot(glm::vec4(v_bar,1.0f)* (q[it->x] + q[it->y - 1]) ,glm::vec4(v_bar,1.0f));
                    it->v_bar = v_bar;

                }
                else if(it->y == argmin.x || it->y == argmin.y){
                    it->y = tmp;
                    glm::mat4 Q_bar = q[it->x] + q[it->y - 1];
                    Q_bar[0][3] = 0;
                    Q_bar[1][3] = 0;
                    Q_bar[2][3] = 0;
                    Q_bar[3][3] = 1;
        
                    // glm::vec4 v_bar = glm::inverse(Q_bar) * glm::vec4(0,0,0,1);

                    glm::mat3 Q_inv = glm::inverse(glm::mat3(Q_bar));
                    glm::vec3 v_bar = Q_inv * glm::vec3(Q_bar[3][0],Q_bar[3][1],Q_bar[3][2]);
                    
                    v_bar = v_bar * glm::vec3(-1.0f);
                    it->cost = glm::dot(glm::vec4(v_bar,1.0f)* (q[it->x] + q[it->y - 1]) ,glm::vec4(v_bar,1.0f));
                    it->v_bar = v_bar;
                }
                if(it->x >= argmin.x) it->x --;
                if(it->x >= argmin.y) it->x --;
                if(it->y >= argmin.x) it->y --;
                if(it->y >= argmin.y) it->y --;
                it++;
            }

            for(auto && it : validPairs){
                if(it.x >= output.Positions.size() - 2)
                    it.x --;
                if(it.y >= output.Positions.size() - 2)
                    it.y --;
            }

            output.Positions.erase(argmin.x + output.Positions.begin());
            output.Positions.erase(argmin.y - 1 + output.Positions.begin());




            // cout<<output.Positions[output.Positions.size() - 1].x<<" "<<output.Positions[output.Positions.size() - 1].y<<" "<<output.Positions[output.Positions.size() - 1].z<<" "<<endl;
            // cout<<output.Positions.size()<<endl;

            //update q
            
            q.erase(argmin.x + q.begin());
            q.erase(argmin.y - 1 + q.begin());

            for(auto it = validPairs.begin(); it != validPairs.end();it ++){
                for(auto jt = it + 1; jt != validPairs.end();){
                    if(it->x == jt ->x && it->y == jt->y) {
                        validPairs.erase(jt);
                    }
                    else jt ++;
                }
            }

            //update valid pairs

        }
        
        

=======
    }

    /******************* 3. Mesh Simplification *****************/
    void SimplifyMesh(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, float valid_pair_threshold, float simplification_ratio) {
        // your code here
>>>>>>> c1db74a9bd05d4381ce103d6e5f5baffd5222371
    }

    /******************* 4. Mesh Smoothing *****************/
    void SmoothMesh(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, std::uint32_t numIterations, float lambda, bool useUniformWeight) {
        // your code here
<<<<<<< HEAD

        output.Positions = input.Positions;
        output.Indices = input.Indices;
        
        for(int iter = 0; iter < numIterations; iter++){
            vector<glm::vec3> tmp_pos = output.Positions;

            DCEL links;
            links.AddFaces(output.Indices);
            if(! links.IsValid())
                return;
            for(std::size_t i = 0; i < output.Positions.size(); i++){
                DCEL::Vertex v = links.GetVertex(i);
                if(tmp_pos[i] != output.Positions[i]) continue;
                glm::vec3 new_v(0.0f);
                float co_sum = 0.0;
                for(auto && u : v.GetNeighbors()){
                    if(useUniformWeight){
                        co_sum += 1.0f;
                        new_v += output.Positions[u];
                    }
                    else{
                        const DCEL::HalfEdge  * e = NULL;
                        for(auto && k : links.GetEdges()){
                            if((k->To() == i && k->From() == u) || (k->To() == u && k->From() == i)){
                                e = k;
                                break;
                            }
                        }
                        glm::vec3 a_pos = output.Positions[e->OppositeVertex()];
                        glm::vec3 x_pos = output.Positions[i];
                        glm::vec3 y_pos = output.Positions[u];

                        glm::vec3 v1 = x_pos - a_pos;
                        glm::vec3 v2 = y_pos - a_pos;


                        float cos_alpha = glm::dot(v1,v2);


                        float cos_beta = glm::dot(v1,v2); 
                        float w;           

                        a_pos = output.Positions[e->PairOppositeVertex()];
                        v1 = x_pos - a_pos;
                        v2 = y_pos - a_pos;

                        cos_beta = glm::dot(v1,v2);

                        w = cos_alpha/sqrt(1-cos_alpha*cos_alpha) +
                            cos_beta/sqrt(1-cos_beta*cos_beta) ;
                        
                        co_sum += w;
                        new_v += w * output.Positions[u];
                    }
                }
                new_v = new_v * glm::vec3(1.0f/co_sum);

                tmp_pos[i]= glm::vec3(1-lambda) * output.Positions[i] + glm::vec3(lambda) * new_v;
            }

            for(std::size_t i = 0; i < output.Positions.size(); i++)
                output.Positions[i] = tmp_pos[i];

        }
    }

    /******************* 5. Marching Cubes *****************/

    glm::vec3 unit(int i){
        if(i == 0) return glm::vec3(1,0,0);
        if(i == 1) return glm::vec3(0,1,0);
        if(i == 2) return glm::vec3(0,0,1);
        else return glm::vec3(0);
    }

    void MarchingCubes(Engine::SurfaceMesh & output, const std::function<float(const glm::vec3 &)> & sdf, const glm::vec3 & grid_min, const float dx, const int n) {
        // your code here
        glm::vec3 vertexPos = grid_min;
        int totVertex = 0;

        for(int xi = 0; xi < n; xi++)
            for(int xj = 0; xj < n; xj++)
                for(int xk = 0; xk < n; xk++){
                    uint16_t Idx = 0;
                    vertexPos  = grid_min +glm::vec3(xi * dx, xj * dx, xk * dx);
                    vector<glm::vec3> vertexPosI, vertexNormal;
                    vector<float> vertexSdf;
                    for(int p = 0; p <= 7; p++){
                        glm::vec3 tmpPos = vertexPos + glm::vec3((p&1) * dx, ((p>>1) & 1) * dx, ( (p>>2) & 1) * dx);
                        if(sdf(tmpPos) <= 0)
                            Idx |= (1<<p);
                    }
                    if(Idx == 0 || Idx == 255) continue;
                    

                    for(int p = 0; c_EdgeOrdsTable[Idx][p] != -1; p += 3){
                        for(int l = 0; l <= 2; l++){
                            int j = c_EdgeOrdsTable[Idx][p + l];
                            glm::vec3 stPos = vertexPos + glm::vec3(dx * (j & 1)) * unit(((j >> 2) + 1) % 3)
                                            + glm::vec3(dx * ((j >> 1) & 1)) * unit(((j >> 2) + 2) % 3);
                            glm::vec3 edPos = stPos + unit(j>>2) * glm::vec3(dx,dx,dx);
                            float stSdf = sdf(stPos);
                            float edSdf = sdf(edPos);

                            glm::vec3 tmpPos;

                            if(edSdf == stSdf) 
                                tmpPos = glm::vec3(0.5f) * (stPos + edPos);
                            else tmpPos = stPos + glm::vec3(-stSdf/(edSdf - stSdf)) * (edPos - stPos);
                            auto v = find(output.Positions.begin(),output.Positions.end(), tmpPos);
                            if(v == output.Positions.end()){
                                output.Indices.push_back(totVertex++);
                                output.Positions.push_back(tmpPos);
                            }
                            else output.Indices.push_back(v - output.Positions.begin());
                            

                        }
                    }


                }

    }
} // namespace VCX::Labs::GeometryProcessing

=======
    }

    /******************* 5. Marching Cubes *****************/
    void MarchingCubes(Engine::SurfaceMesh & output, const std::function<float(const glm::vec3 &)> & sdf, const glm::vec3 & grid_min, const float dx, const int n) {
        // your code here
    }
} // namespace VCX::Labs::GeometryProcessing
>>>>>>> c1db74a9bd05d4381ce103d6e5f5baffd5222371
