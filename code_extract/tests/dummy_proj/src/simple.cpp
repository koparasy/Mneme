#include "simple.h"
#include "A.h"

#include <cmath>
#include <memory>

int a = 10; 

class B {
    int b1;
    public: 
    int fb1() { return b1; }
};

int func(int a, int b) {
    int c = a + b;
    return c * c;
}

int o = func(1, 2);

int func2(int k) {
    return x + a;
}

int func3(int a) {
    return ::a * 42;
}

int func4(int k) {
    static int y = 1;
    int b = x + a;
    y += k + b + o;
    return y;
}

int func5(int k) {
    int m = func(1, 2) + func2(m);
    return m;
}

int func6(int k) {
    A obj1;
    B obj2;
    int m = obj1.f1() + A::f2() + obj2.fb1();
    return m;
}

int func7() {
    A* obj = new A();
    B* obj2 = new B();
    B** obj2_ptr = &obj2;
    return obj->f1() + (*obj2_ptr)->fb1();
}

int func8(B obj) {
    return obj.fb1() + A::f2();
}

double func9(int m) {
    return std::cos(m);
}

int func10() {
    std::unique_ptr<A> obj(new A());
    return obj->f1();
}

int main() {
    // init_gpu_memory(char const* arg1, char const* arg2) -- call to record replay to init all params.
    int a =1;
    int b = 2;
    func2(1);
    // verify_ouput() -- check the output of the pulled function, use record replay to check against recorded output

    // // inject prologue
    // register_value(&a);
    // register_value(&b);
    // func(a, b);
    // // inject epilogue
    // func4();
    return 0;
}