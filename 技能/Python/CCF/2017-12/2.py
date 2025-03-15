class node:
    def __init__(self, number, state):
        self.number = number + 1
        self.next = None

data = list(map(int, input().split()))
n = data[0]
k = data[1]
head = node(0, True)
current = head
for i in range(1, n):
    new_node = node(i, True)
    current.next = new_node
    current = new_node
current.next = head
pre = current
current = head
count = 1
while current.next != current:
    if count % k == 0 or count % 10 == k:
        pre.next = current.next  
    else:
        pre = current  
    count += 1

    current = current.next
print(current.number)