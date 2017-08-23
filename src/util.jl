"""
Helper macro to generate non-inplace versions of functions.

As an example, given a function elu!(out, xs, alpha), call:

    `@outplace elu(xs, alpha)`

This will generate a function definition equivalent to:

    `elu(xs, alpha) = elu!(similar(xs), xs, alpha)`
"""
macro outplace(f)
    # Generate the outplace version
    fname = esc(Symbol(f.args[1],:!))
    args = esc.(f.args[2:end])
    return quote
        $(esc(f.args[1]))($(args...)) = $(fname)(similar($(args[1])), $(args...))
    end
end

"""
Given function name `foo`, exports `foo`, `foo!`, `∇foo`` and `∇foo!`

Can also accept multiple function names, e.g. `@multiexport foo, bar`
"""
macro multiexport(f)
    function gen_names(n)
        return [n, Symbol(n, :!), Symbol(:∇,n), Symbol(:∇,n,:!)]
    end

    # If we were given only a single thing to export, wrap it in a tuple
    if !(typeof(f) <: Expr)
        f = :(($(f),))
    end
        
    # If we've been given multiple things to export, do them all
    names = []
    for n in f.args
        push!(names, gen_names(n)...)
    end

    return Expr(:export, names...)
end