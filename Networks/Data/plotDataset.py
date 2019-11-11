import matplotlib.pyplot as plt

classListFile = open("../Data/dataset/classList.txt", "r")
classList = classListFile.readlines()
classListFile.close()

classCountList = []
classNameList = []
for i in range(0, len(classList)):
    className = classList[i].split('|')[0]
    classCount = int(classList[i].split('|')[1])
    classCountList.append(classCount)
    classNameList.append(i)


plt.figure(figsize=(20, 3))

plt.bar(classNameList, classCountList)
plt.title('Snake Dataset')
plt.ylabel('Samples')
plt.xticks(classNameList)
plt.xlabel('Class')



plt.show()