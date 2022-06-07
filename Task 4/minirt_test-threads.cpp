#include "minirt/minirt.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <chrono>


using namespace minirt;
using namespace std;

mutex locker;

void initScene(Scene &scene) {
    Color red {1, 0.2, 0.2};
    Color blue {0.2, 0.2, 1};
    Color green {0.2, 1, 0.2};
    Color white {0.8, 0.8, 0.8};
    Color yellow {1, 1, 0.2};

    Material metallicRed {red, white, 50};
    Material mirrorBlack {Color {0.0}, Color {0.9}, 1000};
    Material matteWhite {Color {0.7}, Color {0.3}, 1};
    Material metallicYellow {yellow, white, 250};
    Material greenishGreen {green, 0.5, 0.5};

    Material transparentGreen {green, 0.8, 0.2};
    transparentGreen.makeTransparent(1.0, 1.03);
    Material transparentBlue {blue, 0.4, 0.6};
    transparentBlue.makeTransparent(0.9, 0.7);

    scene.addSphere(Sphere {{0, -2, 7}, 1, transparentBlue});
    scene.addSphere(Sphere {{-3, 2, 11}, 2, metallicRed});
    scene.addSphere(Sphere {{0, 2, 8}, 1, mirrorBlack});
    scene.addSphere(Sphere {{1.5, -0.5, 7}, 1, transparentGreen});
    scene.addSphere(Sphere {{-2, -1, 6}, 0.7, metallicYellow});
    scene.addSphere(Sphere {{2.2, 0.5, 9}, 1.2, matteWhite});
    scene.addSphere(Sphere {{4, -1, 10}, 0.7, metallicRed});
    scene.addSphere(Sphere {{5, 3, 3}, 1, mirrorBlack});//////////////////
    scene.addSphere(Sphere {{1.2, -0.5, 6}, 0.5, matteWhite});////////////

    scene.addLight(PointLight {{-15, 0, -15}, white});
    scene.addLight(PointLight {{1, 1, 0}, blue});
    scene.addLight(PointLight {{0, -10, 6}, red});
    scene.addLight(PointLight {{-15, 1, 6}, red});////////////////////////
    scene.addLight(PointLight {{15, 5, 15}, white});//////////////////////

    scene.setBackground({0.05, 0.05, 0.08});
    scene.setAmbient({0.1, 0.1, 0.1});
    scene.setRecursionLimit(20);

    scene.setCamera(Camera {{0, 0, -20}, {0, 0, 0}});
}

void ElementaryTask(const Scene &scene, Image &image, ViewPlane &viewPlane, pair<int,int> point){
    int x = point.first;
    int y = point.second;
    const auto color = viewPlane.computePixel(scene, x, y, 1);
    image.set(x, y, color);
}

void ThreadWork(const Scene &scene, Image &image, ViewPlane &viewPlane, queue<pair<int,int>> &tasks){
    while(true){
        locker.lock();
        if(tasks.empty()){
            locker.unlock();
            break;
        }
        pair<int,int> currentTask = tasks.front();
        tasks.pop();
        locker.unlock();
        ElementaryTask(scene, image, viewPlane, currentTask);
    }
}


int main(int argc, char **argv) {
    int viewPlaneResolutionX = (argc > 1 ? stoi(argv[1]) : 3000);////
    int viewPlaneResolutionY = (argc > 2 ? stoi(argv[2]) : 3000);////
    int numOfSamples = (argc > 3 ? stoi(argv[3]) : 1);    
    string sceneFile = (argc > 4 ? argv[4] : "");

    Scene scene;
    if (sceneFile.empty()) {
        initScene(scene);
    } else {
        scene.loadFromFile(sceneFile);
    }

    const double backgroundSizeX = 4;
    const double backgroundSizeY = 4;
    const double backgroundDistance = 10;

    const double viewPlaneDistance = 5;
    const double viewPlaneSizeX = backgroundSizeX * viewPlaneDistance / backgroundDistance;
    const double viewPlaneSizeY = backgroundSizeY * viewPlaneDistance / backgroundDistance;

    ViewPlane viewPlane {viewPlaneResolutionX, viewPlaneResolutionY,
                         viewPlaneSizeX, viewPlaneSizeY, viewPlaneDistance};

    Image image(viewPlaneResolutionX, viewPlaneResolutionY); // computed image
    
    //auto start = chrono::high_resolution_clock::now();
    int threadNum = 16;

    vector<thread> workers;
    queue<pair<int,int>> tasks;
    for(int x = 0; x < viewPlaneResolutionX; x++){
        for(int y = 0; y < viewPlaneResolutionY; y++){
            tasks.push({x, y});
        }
    }
    
    auto start = chrono::high_resolution_clock::now();
    for(int i = 0; i < threadNum; i++){
        workers.emplace_back(ThreadWork, ref(scene), ref(image), ref(viewPlane), ref(tasks));
    }

    for(auto &worker: workers){
        worker.join();
    }

    auto end = chrono::high_resolution_clock::now();
    float duration = chrono::duration<float>(end - start).count();
    cout << "Time = " << duration << endl;

    image.saveJPEG("raytracing.jpg");

    return 0;
}





