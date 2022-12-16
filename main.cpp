#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

struct derivatives{
    vector<vector<vector<float>>> weights;
    vector<vector<float>> bias;
    derivatives(vector<vector<vector<float>>> weights_, vector<vector<float>> bias_){
        weights = weights_;
        bias = bias_;
    };
};

class NN {
    private:
    vector<vector<vector<float>>> weights;//layer, float, connections
    vector<vector<float>> bias;
    float relu(float x){
        return max(0.0f,x);
    };
    float derivative_relu(float x){
        return 1;
        return  x < 1;
    }
    float sigmoid(float x){
        return 1/(1+exp(x));
    }
    float dot(vector<float> inputs, vector<float> weights){
        float activation = 0;
        for (float i = 0; i < weights.size(); i++) activation += inputs[i] * weights[i];
        return activation;
    }
    float cost(float a, float b){
        return (a-b)*(a-b);
    }
    float getDerivative(int prevDerivative, int weight){
        return relu(prevDerivative * weight);
    }
    vector<float> add(vector<vector<float>> input){
        vector<float> output;
        int t;
        for (int i = 0; i < input[0].size(); i++){
            t = 0;
            for (int j = 0; j < input.size(); j++){
                t += input[j][i];
            }
            output.push_back(t);
        }
        return output;
    }
    public:
    void floatialize(float inputs, float outputs, float w, float b){//3 2
        vector<vector<float>> newLayer;
        vector<float> curr;
        vector<float> t;
        for (float i = 0; i < outputs; i++) t.push_back(b);
//        for (float i = 0; i < inputs; i++){
//            curr.clear();
//            for (float j = 0; j < outputs; j++){
//                curr.push_back(w);
//            }
//            newLayer.push_back(curr);
//        }
        for (float i = 0; i < inputs; i++) curr.push_back(w);
        for (float j = 0; j < outputs; j++) newLayer.push_back(curr);
        weights.push_back(newLayer);
        bias.push_back(t);
    };
    void create_layer(float size, float w, float b){
        vector<vector<float>> newLayer;
        vector<float> connections;
        vector<float> t;
        for (float i = 0; i < size; i++) t.push_back(b);
        for (float i = 0; i < weights[weights.size() - 1].size(); i++) connections.push_back(w);
        for (float i = 0; i < size; i++) newLayer.push_back(connections);
        weights.push_back(newLayer);
        bias.push_back(t);
    };
    float predict(float input){
        vector<float> prev;//activation of prev layer
        vector<float> curr;
        prev.push_back(input);
        for (float i = 0; i < weights.size(); i++){
            curr.clear();
            for (float j = 0; j < weights[i].size(); j++){
                curr.push_back(relu(dot(prev, weights[i][j])) + bias[i][j]);
            }
            prev = curr;
        }
        return prev[0];
    }
    vector<vector<vector<float>>> backProp(float X, float Y){
        vector<vector<vector<float>>> weightDerivatives;
        vector<vector<float>> layerDerivative;
        vector<vector<float>> biasDerivatives;
        vector<float> prevDerivative;
        vector<float> currDerivative;
        prevDerivative.push_back(predict(X) - Y);
        for (float i = weights.size() - 1; i >= 0; i--){
            layerDerivative.clear();
            for (float j = 0; j < weights[i].size(); j++){
                currDerivative.clear();
                for (int k = 0; k < weights[i][j].size(); k++){
//                    currDerivative.push_back(weights[i][j][k]*relu(prevDerivative[j]));
                    currDerivative.push_back(getDerivative(weights[i][j][k], prevDerivative[j]));
                }
                layerDerivative.push_back(currDerivative);
            }
            prevDerivative = add(layerDerivative);
            biasDerivatives.push_back(prevDerivative);
            weightDerivatives.push_back(layerDerivative);
        }
        return weightDerivatives;
    }
};

int main(){
    //predict test
    NN model;
    model.floatialize(1, 3, 2, 1);
    model.create_layer(3, 1, -2);
    model.create_layer(3, 1, -2);
    cout << model.predict(1);
    model.backProp(1.0, 1.0);
    return 0;
}
//  3
//1 3 3
//  3