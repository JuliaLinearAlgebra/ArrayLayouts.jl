module ArrayLayoutsTests

import ArrayLayouts
using ParallelTestRunner

const init_code = quote
    import Random
    Random.seed!(0)
end

# Start with autodiscovered tests
testsuite = find_tests(pwd())

if "--downstream_integration_test" in ARGS
    delete!(testsuite, "test_aqua")
end

filtered_args = filter(!=("--downstream_integration_test"), ARGS)
# Parse arguments
args = parse_args(filtered_args)

if filter_tests!(testsuite, args)
    delete!(testsuite, "infinitearrays")
end

runtests(ArrayLayouts, args; testsuite, init_code)

end
