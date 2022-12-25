# why use intrusive_ptr?

> https://baptiste-wicht.com/posts/2011/11/boost-intrusive_ptr.html

his post will present the Boost intrusive_ptr and its usage in C++ programming.

Recently, I took some time to optimize the parsing performances of the EDDI Compiler. The parsing phase creates a lot of nodes to fill the Abstract Syntax Tree.

One of the way I found was to replace some shared_ptr by some intrusive_ptr of the Boost library.

It's a faster alternative of shared_ptr. Like its name indicates, it's intrusive. The reference counter is included directely in the managed class, in the contrary of the shared_ptr where the reference counter has to be dynamically allocated to live aside the object. This leads to some performances improvement. Considering memory, the footprint of an intrusive_ptr is the same as the footprint of a raw pointer. This is not the case for the shared_ptr that have a pointer to the object, a pointer to the counter and the counter itself.

For example, if you have a class X:
~~~
class X {
    std::string name;
    int age;
};
~~~
And you use it in your code using a shared_ptr :
~~~
void test(){
    std::shared_ptr<X> x(new X);

    std::cout << x->name << std::endl;
}
~~~
and you want to use an intrusive_ptr, you have first to add a reference counter inside the X class :
~~~
class X {
    std::string name;
    int age;

    long references;
    X() : references(0) {}
};
~~~
And you have to indicate to the intrusive_ptr where the reference counter can be found for this class :
~~~
inline void intrusive_ptr_add_ref(X* x){
    ++x->references;
}

inline void intrusive_ptr_release(X* x){
    if(--x->references == 0) 
        delete x;
}
~~~
And finally you can use the intrusive_ptr to replace your shared_ptr :
~~~
void test(){
    boost::intrusive_ptr<X> x(new X);

    std::cout << x->name << std::endl;
}
~~~

The smart pointer itself can be used exactly the same way as a shared_ptr. If you have several classes that are managed using an intrusive_ptr, you can use a function template to tell Boost that all the reference counter are at the same place :
~~~
template<typename T>
inline void intrusive_ptr_add_ref(T* expr){
++expr->references;
}

template<typename T>
inline void intrusive_ptr_release(T* expr){
if(--expr->references == 0)
delete expr;
}
~~~
As you can see, the pointer is very intrusive and needs some boilerplate code added to your application, but it can leads to some interesting improvements for classes very often dynamically allocated.

There is another advantage in using intrusive_ptr. As the reference counter is stored into the object itself, you can create several intrusive_ptr to the same object without any problem. This is not the case when you use a shared_ptr. Indeed, if you create two shared_ptr to the same dynamically allocated object, they will both have a different references counter and at the end, you will end up with an object being deleted twice.

Of course, there are not only advantages. First of all, you have to declare a field in every classes that you want to manage using an intrusive_ptr and you have to declare functions to manage the reference. Then there are some disadvantages when using this pointer type compared to a shared_ptr :

It's impossible to create a weak_ptr from a intrusive_ptr
Code redundancy, you have to copy the reference counter in every class that you want to use an intrusive_ptr with
You have to provide a function for every types that has to be used with intrusive_ptr (only two functions if you use the template versions of the two functions)
To conclude, the boost::intrusive_ptr can be a good replacement of std::shared_ptr in a performance critical application, but if you have no performances problem, do not use it because it makes your code less clear. If you are concerned by performances when using std::shared_ptr, consider also using std::make_shared to create your pointers, so that the reference counter and the object itself will be allocated at the same place and at the same time, resulting in better performances. Another case where it's interesting to use an intrusive_ptr is when dealing with libraries using a lot of raw pointers, because you can create several intrusive_ptr to the same raw pointer without any problem.

