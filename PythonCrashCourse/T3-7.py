#3-4
arr=["Martin O'Donnell","Ludwig Goransson","Hans Zimmer"]
for a in arr:#此处用到后面的知识。也可以一句一句的输出
	print("Welcome, "+a+"!")
print()
#3-5
print(arr.pop(0)+"is not available.")
arr.append("Henry Jackman")
for a in arr:
	print("Welcome, "+a+"!")
print()
#3-6
print("I've found a bigger table.")
arr.insert(0,"Alan Silvestri")
arr.insert(2,"Ramin Djawadi")
arr.append("Tom Salta")
for a in arr:
	print("Welcome, "+a+"!")
print()
#3-7
while(len(arr)>2):#此处用到后面的知识。也可以一句一句的输出
	print("I'm sorry, "+arr.pop()+", but there's no seat for you.")
for a in arr:
	print("Welcome, "+a+"!")
del arr[0]
del arr[0]
print(arr)