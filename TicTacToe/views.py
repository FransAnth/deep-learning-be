import numpy as np
import torch
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView

from TicTacToe.ai_resources.nn_classes.neural_network import NeuralNetwork

# initialize the model
input_size = 9
hidden_size = 64
output_size = 9

trained_model = NeuralNetwork(input_size, hidden_size, output_size)
trained_model.load_state_dict(
    torch.load(r"TicTacToe/ai_resources/trained_ai_models/tic_tac_toe_nn.pth")
)
trained_model.eval()


def one_hot_encode(board):
    """
    Convert a Tic-Tac-Toe board into a one-hot encoded format.

    Args:
        board (list or numpy array): 1D list of length 9 with values -1 (O), 0 (empty), 1 (X)

    Returns:
        torch.Tensor: 1D tensor of length 27 (one-hot encoded board)
    """
    encoded_board = np.zeros((9, 3))  # 9 cells, 3 possible values

    for i in range(9):
        if board[i] == 1:  # X
            encoded_board[i] = [1, 0, 0]
        elif board[i] == -1:  # O
            encoded_board[i] = [0, 1, 0]
        else:  # Empty
            encoded_board[i] = [0, 0, 1]

    return encoded_board.flatten()


class TicTacToe(APIView):
    def post(self, request):
        board_state = request.data.get("boardState")

        input_data = torch.tensor([board_state], dtype=torch.float32)

        output = trained_model(input_data)
        confidense, prediction = torch.max(output, 1)

        response_data = {
            "confidense": confidense[0],
            "modelPrediction": prediction[0],
        }

        return Response(response_data)
