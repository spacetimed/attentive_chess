# input tokens: [ BOS(1) ][ FEN(77) ][ ACTION(1) ] 79 sequence len 

def move_to_action_id(move: str) -> int:
    """returns an int ≤1968 representing the action-move in UCI format"""
    return 0

def encode(fen: str, move: str) -> list[int]:
    """encodes a FEN string and move into a 79-length sequence vector to feed into the model"""
    return [0]