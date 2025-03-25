import torch
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView

from TicTacToe.ai_resources.nn_classes.neural_network import NeuralNetwork

# initialize the model
input_size = 9
hidden_size = 135
output_size = 9

trained_model = NeuralNetwork(input_size, hidden_size, output_size)
trained_model.load_state_dict(
    torch.load(r"TicTacToe/ai_resources/trained_ai_models/tic_tac_toe_nn.pth")
)
print("model_loaded")
trained_model.eval()


class TicTacToe(APIView):
    def post(self, request):
        board_state = request.data.get("boardState")

        input_data = torch.tensor([board_state], dtype=torch.float32)

        output = trained_model(input_data)
        confidense, prediction = torch.max(output, 1)

        response_data = {
            "confidense": confidense[0],
            "modelPrediction": prediction[0] + 1,
        }

        # Added 1 to the prediction[0] because the training data labels
        # is from 0 - 8 and we need to shift one position to make it
        # 1 - 9 that represents the 1st board up until the 9th board

        return Response(response_data)
