#!/usr/bin/env julia
# Author:Shun Arahata
# kalman filter
using PyPlot
const Ix = 1.9
const Iy = 1.6
const Iz = 2.0
const OBSERVE_STEP =100
rpm2radpersec(rpm) = rpm *2 * π / 60
const ω_s = rpm2radpersec(17)
const STEPNUM = 5000
const STEP = 0.01
const q_std = 0.01
const r_std = 0.01
const Quaternion_ini = [1.0; 0.0; 0.0; 0.0]
const ω_b = [0.1; ω_s + 0.1; 0.0]
const B = [0 0 0;
           0 0 0;
           0 0 0;
           0 0 0;
           1/Ix 0 0;
           0 1/Iy 0;
           0 0 1/Iz]
const Q = [q_std^2 0 0;0 q_std^2 0;0 0 q_std^2]
const R = [r_std^2 0 0;0 r_std^2 0;0 0 r_std^2]
const P_ini = [0.01 0 0 0 0 0 0;
               0 0.01 0 0 0 0 0;
               0 0 0.01 0 0 0 0;
               0 0 0 0.01 0 0 0;
               0 0 0 0 0.01 0 0;
               0 0 0 0 0 0.01 0;
               0 0 0 0 0 0 0.01]

const rng = MersenneTwister(10)

mutable struct Kalman_Filter
    state::Array
    variance::Array
end

function rand_normal(μ, σ)
    #= return a random sample from a normal (Gaussian) distribution
    refering from
    https://www.johndcook.com/blog/2012/02/22/julia-random-number-generation/
    =#
    if σ <= 0.0
        error("standard deviation must be positive")
    end
    u1 = rand(rng)
    u2 = rand(rng)
    r = sqrt( -2.0*log(u1) )
    θ = 2.0*π*u2
    return μ + σ*r*sin(θ)
end

function runge_kutta(f, x, step)
    #=　Fourth-order Runge-Kutta method.

    :param f: differential equation f(x)
     Note: input output must be the same dimension list
    :param x: variable
    :param step: step time
    :return: increment
    =#
    k1 = f(x)
    k2 = f(x + step / 2 * k1)
    k3 = f(x + step / 2 * k2)
    k4 = f(x + step * k3)
    sum = step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return sum
end

@inbounds function make_dcm(q,noise::Int,index::Int)
    #= make dcm vector and (add noise)

    :param x:q
    :param noise: adding noise or not. 0 or 1
    :return :DCM vector
    =#
    q0 = q[1]
    q1 = q[2]
    q2 = q[3]
    q3 = q[4]
    x = [q0 ^ 2 + q1 ^ 2 - q2 ^ 2 - q3 ^ 2 + noise * rand_normal(0,r_std);
        2 * (q1 * q2 + q0 * q3) + noise * rand_normal(0,r_std);
        2 * (q1 * q3 - q0 * q2) + noise * rand_normal(0,r_std)]

    y = [2 * (q1 * q2 - q0 * q3) + noise * rand_normal(0,r_std);
        q0 ^ 2 - q1 ^ 2 + q2 ^ 2 - q3 ^ 2 + noise * rand_normal(0,r_std);
        2 * (q2 * q3 + q0 * q1) + noise * rand_normal(0,r_std)]

    z = [2 * (q1 * q3 + q0 * q2) + noise * rand_normal(0,r_std);
        2 * (q2 * q3 - q0 * q1) + noise * rand_normal(0,r_std);
        q0 ^ 2 - q1 ^ 2 - q2 ^ 2 + q3 ^ 2 + noise * rand_normal(0,r_std)]

    index==1 && return x
    index==2 && return y
    index==3 && return z
    println("Index Error")
end

@inbounds function differential_eq(x,noise::Int)
    q0 = x[1]
    q1 = x[2]
    q2 = x[3]
    q3 = x[4]
    ω_x = x[5]
    ω_y = x[6]
    ω_z = x[7]
    dq0 = 1/2 * (-q1 * ω_x + -q2 * ω_y - q3 * ω_z)
    dq1 = 1/2 * (q0 * ω_x + -q3 * ω_y + q2 * ω_z)
    dq2 = 1/2 * (q3 * ω_x + q0 * ω_y - q1 * ω_z)
    dq3 = 1/2 * (-q2 * ω_x + q1 * ω_y + q0 * ω_z)
    norm = sqrt(dq0^2 + dq1^2 + dq2^2 + dq3^2)
    d_ω_x = (Iy - Iz)/Ix * ω_y * ω_z + noise * rand_normal(0, q_std)/Ix
    d_ω_y = (Iz - Ix)/Iy * ω_z * ω_x + noise * rand_normal(0, q_std)/Iy
    d_ω_z = (Ix - Iy)/Iz * ω_x * ω_y + noise * rand_normal(0, q_std)/Iz
    return [dq0/norm; dq1/norm; dq2/norm; dq3/norm; d_ω_x; d_ω_y; d_ω_z]
end

@inbounds function make_A(filter::Kalman_Filter)
    x = filter.state
    q0 = x[1]
    q1 = x[2]
    q2 = x[3]
    q3 = x[4]
    ω_x = x[5]
    ω_y = x[6]
    ω_z = x[7]
    return  [0 -1/2 * ω_x -1/2 * ω_y -1/2 * ω_z -1/2 * q1 -1/2 * q2 -1/2 * q3;
            1/2 * ω_x 0 1/2 * ω_z -1/2 * ω_y 1/2 * q0 -1/2 * q3 1/2 * q2;
            1/2 * ω_y -1/2 * ω_z 0 1/2 * ω_x 1/2 * q3 1/2 * q0 -1/2 * q1;
            1/2 * ω_z 1/2 * ω_y -1/2 * ω_x 0 -1/2 * q2 1/2 * q1 1/2 * q0;
            0 0 0 0 0 (Iy - Iz)/Ix * ω_z (Iy- Iz)/Ix * ω_y;
            0 0 0 0 (Iz - Ix)/Iy * ω_z 0 (Iz - Ix)/Iy * ω_x;
            0 0 0 0  (Ix-Iy)/Iz * ω_y (Ix-Iy)/Iz * ω_x 0 ]
end

@inbounds function make_H(filter::Kalman_Filter, i)
    #= H matrix

    :param i: index of dcm
    =#
    q0 = filter.state[1]
    q1 = filter.state[2]
    q2 = filter.state[3]
    q3 = filter.state[4]
    if i == 1
        return [2*q0 2*q1 -2*q2 -2*q3 0 0 0;
                2*q3 2*q2 2*q1 2*q0 0 0 0;
                -2*q2 2*q3 -2*q0 2*q1 0 0 0]
    elseif i == 2
        return[-2*q3 2*q2 2*q1 -2*q0 0 0 0;
                2*q0 -2*q1 2*q2 -2*q3 0 0 0;
                2*q1 2*q0 2*q3 2*q2 0 0 0]
    elseif i == 3
        return [2*q2 2*q3 2*q0 2*q1 0 0 0;
                -2*q1 -2*q0 2*q3 2*q2 0 0 0;
                2*q0 -2*q1 -2*q2 2*q3  0 0 0]
    else
        println("Index Error")
    end
end

function normalize_quaternion!(filter::Kalman_Filter)
    q0 = filter.state[1]
    q1 = filter.state[2]
    q2 = filter.state[3]
    q3 = filter.state[4]
    norm = sqrt(q0^2 + q1^2 + q2^2 + q3^2)
    filter.state[1] = q0/norm
    filter.state[2] = q1/norm
    filter.state[3] = q2/norm
    filter.state[4] = q3/norm
end

function predict(filter::Kalman_Filter)
    A = make_A(filter)
    Φ= expm(A*STEP)
    Γ = inv(A) * (Φ-eye(7)) * B
    P = filter.variance
    P = Φ * P*  Φ' + Γ * Q * Γ'
    filter.variance = P
    filter.state += runge_kutta(x -> differential_eq(x, 0), filter.state, STEP)
end

function update(filter::Kalman_Filter, dcm, index::Int)
    #=observation and update step

    :param dcm: dcm vector
    :param index: index of dcm vector
    =#
    M = filter.variance
    H = make_H(filter, index)
    P = M - M * H' * inv(H * M * H' + R) * H * M
    K = P * H' * inv(R)
    dcm_estimated = make_dcm(filter.state, 0,index)
    z = dcm - dcm_estimated
    x̂ = K * z
    filter.variance = P
    filter.state += x̂
    normalize_quaternion!(filter)
end


function main()
    true_value = Array{Float64, 2}(STEPNUM+1,7)
    estimated_value = Array{Float64, 2}(STEPNUM+1,7)
    estimated_variance = Array{Float64,3}(STEPNUM+1,7,7)
    # initial condition
    true_value[1, 1:4] = Quaternion_ini
    true_value[1, 5:7] = ω_b

    estimated_value[1, :] = [rand_normal(0,0.01)  for x in 1:7]'
    estimated_value[1, 1:4] += Quaternion_ini
    estimated_value[1, 5:7] += ω_b
    kalman = Kalman_Filter(estimated_value[1,:],P_ini)
    time = zeros(STEPNUM+1)
    for i in 1:STEPNUM
        time[i+1] = i * STEP
        true_value[i+1, :] = true_value[i, :] +
                    runge_kutta(x -> differential_eq(x, 1),true_value[i,:], STEP)
        predict(kalman)
        if  i % OBSERVE_STEP == OBSERVE_STEP-1
            #observation and update
            index = rand(1:3)
            dcm = make_dcm(true_value[i+1,:],1,index)
            update(kalman, dcm, index)
        end
        estimated_value[i+1, :] = kalman.state
        estimated_variance[i+1, :, :] = kalman.variance
    end
    plot(time, true_value,estimated_value,estimated_variance)
end



function plot(time, x, estimate,variance)

    for i in 5:7
        fig = figure()
        ax = fig[:add_subplot](2,1,1)
        ax[:plot](time, x[:,i], label=L"true")
        ax[:plot](time, estimate[:,i], label=L"estimated")
        ax[:set_xlim]([0, last(time)])
        ax[:set_xlabel]("time [sec]")
        i==5&&ax[:set_ylabel](L"$\omega_x$ [rad/s]")
        i==6&&ax[:set_ylabel](L"$\omega_y$ [rad/s]")
        i==7&&ax[:set_ylabel](L"$\omega_z$ [rad/s]")
        legend(loc = "right", fontsize=15)
        ax2 = fig[:add_subplot](2,1,2)
        ax2[:plot](time, x[:,i]-estimate[:,i],label=L"$\Delta \omega$")
        ax2[:set_xlim]([0, last(time)])
        ax2[:set_xlabel]("time [sec]")
        ax2[:plot](time,sqrt.(variance[:,i,i]), label=L"+$\sigma$")
        ax2[:plot](time,-sqrt.(variance[:,i,i]), label=L"-$\sigma$")
        ax2[:set_ylabel](L"$\Delta \omega$ [rad/s]")
        legend(loc = "right", fontsize=15)
        #PyPlot.plt[:show]()
        i==5&&PyPlot.plt[:savefig]("omega_x.pgf")
        i==6&&PyPlot.plt[:savefig]("omega_y.pgf")
        i==7&&PyPlot.plt[:savefig]("omega_z.pgf")
    end

    for i in 1:4
        fig = figure()
        ax = fig[:add_subplot](2,1,1)
        ax[:plot](time, x[:,i], label=L"true")
        ax[:plot](time, estimate[:,i], label=L"estimated")
        ax[:set_xlim]([0, last(time)])
        ax[:set_xlabel]("time [sec]")
        i==1&&ax[:set_ylabel](L"$q_1$ ")
        i==2&&ax[:set_ylabel](L"$q_2$ ")
        i==3&&ax[:set_ylabel](L"$q_3$ ")
        i==4&&ax[:set_ylabel](L"$q_4$ ")
        legend(loc = "right", fontsize=15)
        ax2 = fig[:add_subplot](2,1,2)
        ax2[:plot](time, x[:,i]-estimate[:,i], label=L"\Delta q")
        ax2[:plot](time, sqrt.(variance[:,i,i]), label=L"+$\sigma$")
        ax2[:plot](time, -sqrt.(variance[:,i,i]), label=L"-$\sigma$")
        ax2[:set_xlim]([0, last(time)])
        ax2[:set_xlabel]("time [sec]")
        legend(loc = "right", fontsize=15)
        # PyPlot.plt[:show]()
        i==1&&PyPlot.plt[:savefig]("q_1.pgf")
        i==2&&PyPlot.plt[:savefig]("q_2.pgf")
        i==3&&PyPlot.plt[:savefig]("q_3.pgf")
        i==4&&PyPlot.plt[:savefig]("q_4.pgf")
    end
end

main()
