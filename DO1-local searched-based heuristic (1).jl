using JuMP, Gurobi, MathOptInterface
using LinearAlgebra
using Graphs
using GLPK
using TravelingSalesman
using TravelingSalesmanExact

module TravelingSalesman
    include("struct.jl")
    include("util.jl")
    include("generator.jl")
    include("mtz.jl")
    include("tree.jl")
    include("dfj.jl")
end

struct TSPData
    pos::Matrix{Int64}
    cost::Matrix{Int64}
end
struct Solution
    from::Vector{Int64}
    to::Vector{Int64}
    cost::Int64
end

# Define a function to generate a random TSPData object
using Random
function generate_random(n::Int64, seed::UInt64 = rand(UInt64))::TSPData
    rng = MersenneTwister(seed)
    pos = rand(rng, 0 : 1000, n, 2)
    cost = 1000000 * ones(Int64, n, n)
    for i in 1 : n - 1, j in i + 1 : n
        d = distance(pos[i, 1], pos[i, 2], pos[j, 1], pos[j, 2])
        cost[i, j] = d
        cost[j, i] = d
    end
    return TSPData(pos, cost)
end

function greedy_tsp(data::TSPData)::Solution
    n = size(data.pos, 1)
    visited = zeros(Bool, n)
    from = zeros(Int64, n)
    to = zeros(Int64, n)
    from[1] = 1
    visited[1] = true
    total_cost = 0
    for i in 2:n
        min_cost = Inf
        min_city = -1
        for j in 1:n
            if !visited[j] && data.cost[from[i-1], j] < min_cost
                min_cost = data.cost[from[i-1], j]
                min_city = j
            end
        end
        from[i] = min_city
        visited[min_city] = true
        total_cost += min_cost
    end
    to[1] = 1
    for i in 2:n-1
        to[i] = from[n-i+2]
    end
    to[n] = 1
    total_cost += data.cost[from[n], 1]
    return Solution(from, to, total_cost)
end


function tsp_local_search(data::TSPData, initial_sol::Solution, max_iter::Int)::Solution
    nnodes = size(data.cost, 1)
    best_tour = initial_sol.from
    best_cost = initial_sol.cost

    for it in 1:max_iter
        # construct a candidate solution using a randomized greedy algorithm
        candidate_tour = zeros(Int, nnodes+1)
        candidate_tour[1] = best_tour[1] # start at the first city visited in the initial solution

        for i in 2:nnodes
            # select next city randomly from the nearest unvisited neighbors
            unvisited = setdiff(best_tour, candidate_tour[1:i-1])
            distances = data.cost[candidate_tour[i-1], unvisited]
            sorted_indices = sortperm(distances)
            j = unvisited[sorted_indices[rand(1:length(sorted_indices))]]
            candidate_tour[i] = j
        end
        candidate_tour[nnodes+1] = best_tour[1] # return to starting city to complete tour
        candidate_cost = tsp_tour_cost(data.cost, candidate_tour)

        # improve candidate solution using 2-opt local search
        candidate_tour, candidate_cost = tsp_2opt(data.cost, candidate_tour, candidate_cost)

        # update best solution if candidate is better
        if candidate_cost < best_cost
            best_tour = copy(candidate_tour)
            best_cost = candidate_cost
        end
    end

    return Solution(best_tour[1:end-1], best_tour[2:end], best_cost)
end



function tsp_tour_cost(cost::Matrix{Int}, tour::Vector{Int})
    return sum(cost[tour[i], tour[i+1]] for i in 1:length(tour)-1)
end


function tsp_2opt(cost::Matrix{Int}, tour::Vector{Int}, tour_cost::Int)
    nnodes = length(tour) - 1
    improved = true
    while improved
        improved = false
        for i = 1:nnodes-2
            for j = i+1:nnodes-1
                new_tour = reverse(tour[i+1:j])
                new_cost = tsp_tour_cost(cost, [tour[1:i]; new_tour; tour[j+1:end]])
                if new_cost < tour_cost
                    tour[i+1:j] = new_tour
                    tour_cost = new_cost
                    improved = true
                end
            end
        end
    end
    return tour, tour_cost
end


function distance(x1::Int64, y1::Int64, x2::Int64, y2::Int64)::Int64
    return ceil(Int64, sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)))
end

data = generate_random(20,UInt(5))
initial_sol = greedy_tsp(data)
max_iter = 1000
best_sol = tsp_local_search(data, initial_sol, max_iter)
println("initial solution:", initial_sol)
println("Best tour found: ", best_sol.from, "->", best_sol.to)
println("Cost: ", best_sol.cost)

