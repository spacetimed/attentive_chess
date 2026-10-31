from utils import bagz

from apache_beam import coders

# from searchless_chess/src/bagz.py
CODERS = {
    'fen': coders.StrUtf8Coder(),
    'move': coders.StrUtf8Coder(),
    'count': coders.BigIntegerCoder(),
    'win_prob': coders.FloatCoder(),
}
CODERS['state_value'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['win_prob'],
))
CODERS['action_value'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['move'],
    CODERS['win_prob'],
))
CODERS['behavioral_cloning'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['move'],
))

# load features
reader = bagz.BagFileReader("../data/action_value-00000-of-02148_data.bag")
print(len(reader))

# print first feature
fen, move, win_prob = CODERS["action_value"].decode(reader[0])
print(fen, move, win_prob)

# try sample pass