#include <iostream>
#include <list>
#include <map>
#include <set>
#include <unordered_set>

#include <glm/gtc/matrix_inverse.hpp>
#include <spdlog/spdlog.h>

#include "Labs/2-GeometryProcessing/DCEL.hpp"
#include "Labs/2-GeometryProcessing/tasks.h"


namespace VCX::Labs::GeometryProcessing {

#include "Labs/2-GeometryProcessing/marching_cubes_table.h"
using namespace std;

    /******************* 1. Mesh Subdivision *****************/
    void SubdivisionMesh(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, std::uint32_t numIterations) {
        // your code here
        DCEL links;
        links.AddFaces(input.Indices);
        if(! links.IsValid())
            return;

        std::uint32_t curIteration = 0;
        while(curIteration <= numIterations){
            curIteration ++;
            // vector<DCEL::Vertex> new_vertex;
            for(DCEL::Triangle const * f : links.GetFaces()){
                for(int i = 0; i <= 2; ++i){
                    glm::vec3 vertexPos(0,0,0);
                    HalfEdge* ed = f->Edges(i);
                    if(f->HasOppositeFace(i)){
                        v = glm::vec3(0.5,0.5,0.5) * (links.GetVertex(ed->To()) + links.GetVertex(ed->To()));
                    }
                    else{
                        // v
                        v = glm::vec3(0.375,0.375,0.375) * (links.GetVertex(ed->To()) + links.GetVertex(ed->To()));
                        
                        v += glm::vec3(0.125,0.125,0.125) * (links.GetVertex(ed -> OppoistVertex()) + links.GetVertex(ed -> PairOppoistVertex()))
                    }
                }
            }

            for(std::size_t i = 0; i < input.Positions.size(); i++){
                DCEL::Vertex v = links.GetVertex(i);
                // int cnt = 0;
                double u;
                int n = v.GetNeighbours().size();
                if( n == 3) u = 3f/16;
                else u = 3/((double)n * 8);
                glm::vec3 newPos(0,0,0);
                for(auto && a : v.GetNeighbours()){
                    newPos = newPos + glm::vec3(u) * a.//;    
                }
                v = (1 - n*u) * v + newPos;
            }

            for(auto && a : links.GetFaces()){
                

            }

            
        }
        

    }

    /******************* 2. Mesh Parameterization *****************/
    void Parameterization(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, const std::uint32_t numIterations) {
        // your code here
        DCEL links;
        
        links.AddFaces(input.Indices);
        if(! links.IsValid())
            return;
        for(std::size_t i = 0; i < input.Positions.size(); ++i){
            DCEL::Vertex v = links.GetVertex(i);
            output.TexCoor.push_back(v);
            if(v.IsSide()){
                output.TexCoor[i] = v;
            }
            else 
        }



    }

    /******************* 3. Mesh Simplification *****************/
    void SimplifyMesh(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, float valid_pair_threshold, float simplification_ratio) {
        // your code here
    }

    /******************* 4. Mesh Smoothing *****************/
    void SmoothMesh(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, std::uint32_t numIterations, float lambda, bool useUniformWeight) {
        // your code here
    }

    /******************* 5. Marching Cubes *****************/
    void MarchingCubes(Engine::SurfaceMesh & output, const std::function<float(const glm::vec3 &)> & sdf, const glm::vec3 & grid_min, const float dx, const int n) {
        // your code here
    }
} // namespace VCX::Labs::GeometryProcessing
