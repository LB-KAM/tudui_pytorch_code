class Person:
    def __call__(self, name):
        print("__call__hello "+name)

    def hello(self, name):
        print("hello "+name)

# 通过控制台输出我们可以看到，__call__方法可以不适用.menthod()的形式去调用，可以直接使用对象名(args)的形式去调用，逻辑上更加方便一些
person = Person()
person("zhangsan")
person.hello("lisi")