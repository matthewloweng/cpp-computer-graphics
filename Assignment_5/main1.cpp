// C++ include
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
// Utilities for the Assignment
#include "raster.h"
#include <gif.h>
#include <fstream>
#include <Eigen/Geometry>
// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"
using namespace std;
using namespace Eigen;

//Image height
const int H = 480;

//Camera settings
const double near_plane = 1.5; //AKA focal length
const double far_plane = near_plane * 100;
const double field_of_view = 0.7854; //45 degrees
const double aspect_ratio = 1.5;
const bool is_perspective = false;
const Vector3d camera_position(0, 0, 3);
const Vector3d camera_gaze(0, 0, -1);
const Vector3d camera_top(0, 1, 0);

//Object
const string data_dir = DATA_DIR;
const string mesh_filename(data_dir + "bunny.off");
MatrixXd vertices;
MatrixXi facets;

//Material for the object
const Vector3d obj_diffuse_color(0.5, 0.5, 0.5);
const Vector3d obj_specular_color(0.2, 0.2, 0.2);
const double obj_specular_exponent = 256.0;

//Lights
std::vector<Vector3d> light_positions;
std::vector<Vector3d> light_colors;
//Ambient light
const Vector3d ambient_light(0.3, 0.3, 0.3);

//Fills the different arrays
void setup_scene(){
    //Loads file
    std::ifstream in(mesh_filename);
    if (!in.good()){
        std::cerr << "Invalid file " << mesh_filename << std::endl;
        exit(1);
    }
    std::string token;
    in >> token;
    int nv, nf, ne;
    in >> nv >> nf >> ne;
    vertices.resize(nv, 3);
    facets.resize(nf, 3);
    for (int i = 0; i < nv; ++i){
        in >> vertices(i, 0) >> vertices(i, 1) >> vertices(i, 2);
    }
    for (int i = 0; i < nf; ++i){
        int s;
        in >> s >> facets(i, 0) >> facets(i, 1) >> facets(i, 2);
        assert(s == 3);
    }

    //Lights
    light_positions.emplace_back(8, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(6, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(4, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(2, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(0, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(-2, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(-4, 8, 0);
    light_colors.emplace_back(16, 16, 16);
}

void build_uniform(UniformAttributes &uniform){
    // setup uniform

    // Create U, V, and W as camera components
    Vector3d w = -camera_gaze / camera_gaze.norm();
    Vector3d u = camera_top.cross(w) / (camera_top.cross(w)).norm();
    Vector3d v = w.cross(u);
    
    
    //Setup Camera with position and direction.
    Vector4d u4 ( u(0), u(1), u(2), 0);
    Vector4d v4 ( v(0), v(1), v(2), 0);
    Vector4d w4 ( w(0), w(1), w(2), 0 );
  
    Vector4d e4 ( camera_position(0), camera_position(1), camera_position(2), 1);

    // compute the camera transformation
    MatrixXd Camera_Matrix;
    Camera_Matrix.resize(4, 4);
    Camera_Matrix.col(0) = u4;
    Camera_Matrix.col(1) = v4;
    Camera_Matrix.col(2) = w4;
    Camera_Matrix.col(3) = e4;
    Camera_Matrix = Camera_Matrix.inverse();

    // setup projection matrix
    double l,b,n,r,t,f;

    t = tan(field_of_view / 2.0f) * near_plane;
    r = t * aspect_ratio; 

    l = -r;
    b = -t;
    n = -near_plane;
    f = -far_plane;

    MatrixXd Orth_Matrix;
    Orth_Matrix.resize(4,4);

    Orth_Matrix <<  Vector4d(2/(r-l), 0, 0, -(r+l)/(r-l)), 
                    Vector4d(0, 2/(t-b), 0, -(t+b) / (t-b)), 
                    Vector4d(0, 0, 2 / (n-f), -(n+f)/(n-f)), 
                    Vector4d(0, 0, 0, 1);
    Orth_Matrix.transposeInPlace();

    

    Matrix4d P;
    P <<    Vector4d(n, 0, 0, 0), 
            Vector4d(0, n, 0, 0),
            Vector4d(0, 0, n+f, -f*n),
            Vector4d(0, 0, 1, 0);
    P.transposeInPlace();

    if (is_perspective){
        // setup prespective camera
        uniform.view = (Orth_Matrix * P * Camera_Matrix);
    }else{
        uniform.view = (Orth_Matrix * Camera_Matrix);
    }
}

void simple_render(Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer){
    
    UniformAttributes uniform;
    build_uniform(uniform);
    Program prgm;

    prgm.VertexShader = [](const VertexAttributes &vertexAtr, const UniformAttributes &uniform) {
        VertexAttributes transformed_Vertex;
        transformed_Vertex.position = uniform.view * vertexAtr.position; 
        return transformed_Vertex;
    };

    prgm.FragmentShader = [](const VertexAttributes &vertexAtr, const UniformAttributes &uniform) {
        return FragmentAttributes(1, 0, 0);
    };

    prgm.BlendingShader = [](const FragmentAttributes &fragmentAtr, const FrameBufferAttributes &previous) {
        return FrameBufferAttributes(fragmentAtr.color[0] * 255, fragmentAtr.color[1] * 255, fragmentAtr.color[2] * 255, fragmentAtr.color[3] * 255);
    };

    std::vector<VertexAttributes> vertex_attributes;
    // build the vertex attributes from vertices and facets
        for(int i = 0; i < facets.rows(); i++) {
        Vector3i triangle = facets.row(i);

        // loop over vertices in triangle
        for(int j = 0; j < 3; j++) {
            // get vertex of triangle
            Vector3d v = vertices.row(triangle[j]);
            // add to vertex attributes
            vertex_attributes.push_back(VertexAttributes(v[0], v[1], v[2]));
        }
    }
    rasterize_triangles(prgm, uniform, vertex_attributes, frameBuffer);
}

Matrix4d compute_rotation(const double alpha){
    // Compute the rotation matrix of angle alpha on the y axis around the object barycenter
    Matrix4d res;
    res <<  Vector4d(cos(alpha), 0, -sin(alpha), 0), 
            Vector4d(0, 1, 0, 0),
            Vector4d(sin(alpha), 0, cos(alpha), 0), 
            Vector4d(0, 0, 0, 1);
    
    return res;
}

void wireframe_render(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer){

    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;

    Matrix4d trafo = compute_rotation(alpha);
    uniform.transform = trafo;

    program.VertexShader = [](const VertexAttributes &vertexAtr, const UniformAttributes &uniform) {

        VertexAttributes transformed;
        transformed.position = uniform.view * (uniform.transform * vertexAtr.position);
        return transformed;
    };

    program.FragmentShader = [](const VertexAttributes &vertexAtr, const UniformAttributes &uniform) {

        return FragmentAttributes(1, 0, 0);
    };

    program.BlendingShader = [](const FragmentAttributes &fragmentAtr, const FrameBufferAttributes &previous) {

        return FrameBufferAttributes(fragmentAtr.color[0] * 255, fragmentAtr.color[1] * 255, fragmentAtr.color[2] * 255, fragmentAtr.color[3] * 255);
    };

    std::vector<VertexAttributes> vertex_attributes;

    // generate the vertex attributes for the edges and rasterize the lines
    // use the transformation matrix
    for (int i = 0; i < facets.rows(); i++) {
        Vector3i r = facets.row(i);
        for (int j = 0; j < 3; j ++) {
            VertexAttributes v(vertices(r(j), 0), vertices(r(j), 1), vertices(r(j), 2));
            vertex_attributes.push_back(v);
            int k = (j == 2) ? 0 : j + 1;
            VertexAttributes b(vertices(r(k), 0), vertices(r(k), 1), vertices(r(k), 2));
            vertex_attributes.push_back(b);
        }
    }

    rasterize_lines(program, uniform, vertex_attributes, 0.5, frameBuffer);
}

void get_shading_program(Program &program){
   program.VertexShader = [](const VertexAttributes &vertexAtr, const UniformAttributes &uniform) {
        // transform the position and the normal
        // compute the correct lighting
        VertexAttributes transformed;

        transformed.position = uniform.view * (uniform.transform * vertexAtr.position);
        transformed.normal = uniform.view * vertexAtr.normal;
        const Vector3d pos ( transformed.position(0), transformed.position(1), transformed.position(2) );
        Vector4d n_transform = uniform.transform * vertexAtr.normal;
        const Vector3d norm ( n_transform(0), n_transform(1), n_transform(2) );
        Vector3d v = (camera_position - pos).normalized();
        Vector3d lights_color(0,0,0);

        for (int i = 0; i < light_positions.size(); i ++) {
            const Vector3d &light_position = light_positions[i];
            const Vector3d &light_color = light_colors[i];

            const Vector3d Li = (light_position - pos).normalized();

            const Vector3d diffuse = obj_diffuse_color * std::max(Li.dot(norm), 0.0);

            Vector3d l = (light_position - pos).normalized();
            Vector3d h = (v + l) / (v + l).norm();
            const Vector3d specular = obj_specular_color * std::pow(h.transpose() * norm, obj_specular_exponent);

            Vector3d D = light_position - pos;
            lights_color += (diffuse + specular).cwiseProduct(light_color) / D.squaredNorm();
        }

        transformed.color = ambient_light + lights_color;
        return transformed;
    }; 

    program.FragmentShader = [](const VertexAttributes &vertexAtr, const UniformAttributes &uniform) {
        // create the correct fragment
 
        FragmentAttributes output(vertexAtr.color(0), vertexAtr.color(1), vertexAtr.color(2));
        if(is_perspective) {
            output.position = vertexAtr.position;
        }else{
            output.position = vertexAtr.position.cwiseProduct(Vector4d(1, 1, -1, 1));
        }
        return output;
    };

    program.BlendingShader = [](const FragmentAttributes &fragmentAtr, const FrameBufferAttributes &previous) {
        // implement the depth check
        if (fragmentAtr.position[2] < previous.depth){
            FrameBufferAttributes out(fragmentAtr.color[0] * 255, fragmentAtr.color[1] * 255, fragmentAtr.color[2] * 255, fragmentAtr.color[3] * 255);
            out.depth = fragmentAtr.position[2];
            return out;
        }else{
            return previous;
        }
    };
}

void flat_shading(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer){
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;
    get_shading_program(program);

    Eigen::Matrix4d trafo = compute_rotation(alpha);
    uniform.transform = trafo;
    std::vector<VertexAttributes> vertex_attributes;
    // compute the normals
    // set material colors
    for (int i = 0; i < facets.rows(); i++) {
        Vector3i r = facets.row(i);
        Vector3d a (vertices(r(0), 0), vertices(r(0), 1), vertices(r(0), 2));
        Vector3d b (vertices(r(1), 0), vertices(r(1), 1), vertices(r(1), 2));
        Vector3d c (vertices(r(2), 0), vertices(r(2), 1), vertices(r(2), 2));
        Vector3d n = ((b - a).cross(c - a)).normalized();
        Vector4d normal (n(0), n(1), n(2), 0);

        for (int j = 0; j < 3; j ++) {
            VertexAttributes v(vertices(r(j), 0), vertices(r(j), 1), vertices(r(j), 2));
            v.normal = normal;
            vertex_attributes.push_back(v);
        }
    }

    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}

void pv_shading(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;
    get_shading_program(program);

    Eigen::Matrix4d trafo = compute_rotation(alpha);
    uniform.transform = trafo;

    // compute the vertex normals as vertex normal average

    std::vector<VertexAttributes> vertex_attributes;
    Eigen::MatrixXd Vector_Normal;
    Vector_Normal.resize(vertices.rows(), 3);
    Vector_Normal.setZero();
    
    for(int i = 0; i < facets.rows(); i ++) {
        Vector3i r = facets.row(i);
        Vector3d a (vertices(r(0), 0), vertices(r(0), 1), vertices(r(0), 2));
        Vector3d b (vertices(r(1), 0), vertices(r(1), 1), vertices(r(1), 2));
        Vector3d c (vertices(r(2), 0), vertices(r(2), 1), vertices(r(2), 2));
        Vector3d n = ((b - a).cross(c - a)).normalized();

        Vector_Normal.row(r(0)) += n;
        Vector_Normal.row(r(1)) += n;
        Vector_Normal.row(r(1)) += n;
    }

    for (int i = 0; i < facets.rows(); i++) {
        Vector3i r = facets.row(i);
        for (int j = 0; j < 3; j ++) {
            VertexAttributes v(vertices(r(j), 0), vertices(r(j), 1), vertices(r(j), 2));
            Vector3d n = Vector_Normal.row(r(j)).normalized();
            Vector4d normal (n(0), n(1), n(2), 0);
            v.normal = normal;
            vertex_attributes.push_back(v);
        }
    }

    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}

int main(int argc, char *argv[])
{
    setup_scene();

    int W = H * aspect_ratio;
    Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> frameBuffer(W, H);
    vector<uint8_t> image;

    //simple render
    simple_render(frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("simple.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);
    frameBuffer = Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic>(W, H);

    //wireframe render
    wireframe_render(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("wireframe.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);
    frameBuffer = Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic>(W, H);

    //Flat shading render
    flat_shading(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("flat_shading.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);
    frameBuffer = Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic>(W, H);

    //pv_shading render
    pv_shading(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("pv_shading.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);
    frameBuffer = Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic>(W, H);

    //Wireframe Gif
    int delay = 4;
    GifWriter g;
    GifBegin(&g, "wireframe.gif", frameBuffer.rows(), frameBuffer.cols(), delay);
    for (float i = 0; i < 2 * M_PI; i += 0.05)
    {
        frameBuffer.setConstant(FrameBufferAttributes());
        wireframe_render(i, frameBuffer);
        framebuffer_to_uint8(frameBuffer, image);
        GifWriteFrame(&g, image.data(), frameBuffer.rows(), frameBuffer.cols(), delay);
    }
    GifEnd(&g);

    //flat shading gif
    GifBegin(&g, "flat_shading.gif", frameBuffer.rows(), frameBuffer.cols(), delay);
    for (float i = 0; i < 2 * M_PI; i += 0.05)
    {
        frameBuffer.setConstant(FrameBufferAttributes());
        flat_shading(i, frameBuffer);
        framebuffer_to_uint8(frameBuffer, image);
        GifWriteFrame(&g, image.data(), frameBuffer.rows(), frameBuffer.cols(), delay);
    }
    GifEnd(&g);

    //pv_shading Gif
    GifBegin(&g, "pv_shading.gif", frameBuffer.rows(), frameBuffer.cols(), delay);
    for (float i = 0; i < 2 * M_PI; i += 0.05)
    {
        frameBuffer.setConstant(FrameBufferAttributes());
        pv_shading(i, frameBuffer);
        framebuffer_to_uint8(frameBuffer, image);
        GifWriteFrame(&g, image.data(), frameBuffer.rows(), frameBuffer.cols(), delay);
    }
    GifEnd(&g);

    return 0;
}
//end