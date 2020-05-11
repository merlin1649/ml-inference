#import flask
from flask import Flask, request
import grpc
#from tensorrtserver.api import api_pb2
from tensorrtserver.api import grpc_service_pb2
from tensorrtserver.api import grpc_service_pb2_grpc
#import tensorrtserver.api.model_config_pb2 as model_config

# Create gRPC stub for communicating with the server
channel = grpc.insecure_channel('35.225.226.0:8001')
grpc_stub = grpc_service_pb2_grpc.GRPCServiceStub(channel)


app = Flask(__name__)

@app.route('/infer', methods=["POST"])
def infer():
    r = grpc_service_pb2.InferRequest.FromString(request.data)
    return grpc_stub.Infer(r).SerializeToString()

@app.route('/status', methods=["POST"])
def status():
    r = grpc_service_pb2.StatusRequest.FromString(request.data)
    return grpc_stub.Status(r).SerializeToString()

if __name__ == '__main__':
    app.run()
