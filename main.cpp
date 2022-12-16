#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

struct Derivatives{
    vector<vector<vector<float>>> weights;
    vector<vector<float>> bias;
    Derivatives(vector<vector<vector<float>>> weights_, vector<vector<float>> bias_){
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
    float dot(vector<float> inputs, vector<float> weights){
        float activation = 0;
        for (float i = 0; i < weights.size(); i++) activation += inputs[i] * weights[i];
        return activation;
    }
    float cost(float a, float b){
        return (a-b)*(a-b);
    }
    float getDerivative(float prevDerivative, float weight){
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
    void initialize(float inputs, float outputs, float w, float b){//3 2
        vector<vector<float>> newLayer;
        vector<float> curr;
        vector<float> t;
        for (int i = 0; i < outputs; i++) t.push_back(b);
        for (int i = 0; i < inputs; i++) curr.push_back(w);
        for (int j = 0; j < outputs; j++) newLayer.push_back(curr);
        weights.push_back(newLayer);
        bias.push_back(t);
    };
    void create_layer(float size, float w, float b){
        vector<vector<float>> newLayer;
        vector<float> connections;
        vector<float> t;
        for (int i = 0; i < size; i++) t.push_back(b);
        for (int i = 0; i < weights[weights.size() - 1].size(); i++) connections.push_back(w);
        for (int i = 0; i < size; i++) newLayer.push_back(connections);
        weights.push_back(newLayer);
        bias.push_back(t);
    };
    float predict(float input){
        vector<float> prev;//activation of prev layer
        vector<float> curr;
        prev.push_back(input);
        for (int i = 0; i < weights.size(); i++){
            curr.clear();
            for (int j = 0; j < weights[i].size(); j++){
                curr.push_back(relu(dot(prev, weights[i][j])) + bias[i][j]);
            }
            prev = curr;
        }
        return prev[0];
    }
    Derivatives backProp(float X, float Y){
        vector<vector<vector<float>>> weightDerivatives;
        vector<vector<float>> layerDerivative;
        vector<vector<float>> biasDerivatives;
        vector<float> prevDerivative;
        vector<float> currDerivative;
        prevDerivative.push_back(predict(X) - Y);
        biasDerivatives.push_back(prevDerivative);
        for (int i = weights.size() - 1; i >= 0; i--){
            layerDerivative.clear();
            for (int j = 0; j < weights[i].size(); j++){
                currDerivative.clear();
                for (int k = 0; k < weights[i][j].size(); k++){
//                    currDerivative.push_back(weights[i][j][k]*relu(prevDerivative[j]));
                    currDerivative.push_back(getDerivative(prevDerivative[j], weights[i][j][k]));
                }
                layerDerivative.push_back(currDerivative);
            }
            prevDerivative = add(layerDerivative);
            biasDerivatives.push_back(prevDerivative);
            weightDerivatives.push_back(layerDerivative);
        }
        reverse(weightDerivatives.begin(), weightDerivatives.end());
        reverse(biasDerivatives.begin(), biasDerivatives.end());
        biasDerivatives.pop_back();
        return Derivatives(weightDerivatives, biasDerivatives);
    };
    void fit(float X, float Y, float lr, int epochs){
        for (int epoch = 0; epoch < epochs; epoch++){
            Derivatives derivatives = backProp(X, Y);
            for (int i = 0; i < weights.size(); i++){
                for (int j = 0; j < weights[i].size(); j++){
                    for (int k = 0; k < weights[i][j].size(); k++){
                        weights[i][j][k] -= lr*derivatives.weights[i][j][k];
                    }
                }
            }
            for (int i = 0; i < bias.size(); i++){
                for (int j = 0; j < bias[i].size(); j++){
                    bias[i][j] -= lr*derivatives.bias[i][j];
                }
            }
        }
    }
};

int main(){
    //predict test
    NN model;
    model.initialize(1, 3, 2, 1);
    model.create_layer(1, 1, -2);
    cout << model.predict(1) << " ";
    model.fit(1.0, 1.0, 0.01, 1000);
    model.backProp(1.0, 1.0);
    cout << model.predict(1.0);
    return 0;
}
//  3
//1 3 3
//  3