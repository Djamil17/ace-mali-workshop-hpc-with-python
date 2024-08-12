import marshal as marsh
import pickle as pick

message = "hello world!"
pickled_message = pick.dumps(message)
marshalled_message = marsh.dumps(message)

print(pickled_message)
print(marshalled_message)
