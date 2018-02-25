#!/usr/bin/env julia
# Author:Shun Arahata
# kalman filter
using PyPlot
const Ix = 1.9
const Iy = 1.6
const Iz = 2.0
rpm2radpersec(rpm) = rpm *2 * pi / 60
const omega_s = rpm2radpersec(17)
const STEPNUM = 10000
const STEP = 0.01
const q_std = 0.01
const r_std = 0.01
Quaternion_ini = [1.0; 0.0; 0.0; 0.0]
omega_b = [0.1; omega_s + 0.1; 0.0]
B = [0 0 0;0 0 0;0 0 0;0 0 0;1/Ix 0 0;0 1/Iy 0;0 0 1/Iz]
Q = [q_std^2 0 0;0 q_std^2 0;0 0 q_std^2]
R = [r_std^2 0 0;0 r_std^2 0;0 0 r_std^2]
P_ini = [0.01 0 0 0 0 0 0;0 0.01 0 0 0 0 0;0 0 0.01 0 0 0 0;0 0 0 0.01 0 0 0;
        0 0 0 0 0 0 0.01;0 0 0 0 0 0.01 0;0 0 0 0 0 0 0.01]

mutable struct Kalman_Filter
    state::Array
    variance::Array
end

function rand_normal(mean, stdev)
    #= return a random sample from a normal (Gaussian) distribution
    refering from
    https://www.johndcook.com/blog/2012/02/22/julia-random-number-generation/
    =#
    if stdev <= 0.0
        error("standard deviation must be positive")
    end
    u1 = rand()
    u2 = rand()
    r = sqrt( -2.0*log(u1) )
    theta = 2.0*pi*u2
    return mean + stdev*r*sin(theta)
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

function make_dcm(q,noise::Int)
    #= make dcm vector and (add noise)

    :param x:q
    :param noise: adding noise or not. 0 or 1
    :return :2 DCM vector
    =#
    q0 = q[1]
    q1 = q[2]
    q2 = q[3]
    q3 = q[4]
    x = [q0 ^ 2 + q1 ^ 2 - q2 ^ 2 - q3 ^ 2 + noise * rand_normal(0,r_std);
        2 * (q1 * q2 - q0 * q3) + noise * rand_normal(0,r_std);
        2 * (q1 * q3 + q0 * q2) + noise * rand_normal(0,r_std)]

    y = [2 * (q1 * q2 + q0 * q3) + noise * rand_normal(0,r_std);
        q0 ^ 2 - q1 ^ 2 + q2 ^ 2 - q3 ^ 2 + noise * rand_normal(0,r_std);
        2 * (q2 * q3 - q0 * q1) + noise * rand_normal(0,r_std)]

    z = [2 * (q1 * q3 - q0 * q2) + noise * rand_normal(0,r_std);
        2 * (q2 * q3 + q0 * q1) + noise * rand_normal(0,r_std);
        q0 ^ 2 - q1 ^ 2 - q2 ^ 2 + q3 ^ 2 + noise * rand_normal(0,r_std)]

    return [x, y, z]
end

function differential_eq(x,noise::Int)
    q0 = x[1]
    q1 = x[2]
    q2 = x[3]
    q3 = x[4]
    omega_x = x[5]
    omega_y = x[6]
    omega_z = x[7]
    dq0 = 1/2 * (-q1 * omega_x + -q2 * omega_y - q3 * omega_z)
    dq1 = 1/2 * (q0 * omega_x + -q3 * omega_y + q2 * omega_z)
    dq2 = 1/2 * (q3 * omega_x + q0 * omega_y - q1 * omega_z)
    dq3 = 1/2 * (-q2 * omega_x + q1 * omega_y + q0 * omega_z)
    norm = sqrt(dq0^2 + dq1^2 + dq2^2 + dq3^2)
    new_omega_x = (Iy - Iz)/Ix * omega_y * omega_z + noise * rand_normal(0, q_std)/Ix
    new_omega_y = (Iz - Ix)/Iy * omega_z * omega_x + noise * rand_normal(0, q_std)/Iy
    new_omega_z = (Ix - Iy)/Iz * omega_x * omega_y + noise * rand_normal(0, q_std)/Iz
    return [dq0/norm; dq1/norm; dq2/norm; dq3/norm; new_omega_x; new_omega_y; new_omega_z]
end

function make_A(filter::Kalman_Filter)
    x = filter.state
    q0 = x[1]
    q1 = x[2]
    q2 = x[3]
    q3 = x[4]
    omega_x = x[5]
    omega_y = x[6]
    omega_z = x[7]
    return  [0 -1/2 * omega_x -1/2 * omega_y -1/2 * omega_z -1/2 * q1 -1/2 * q2 1/2 * q3;
            1/2 * omega_x 0 1/2 * omega_z -1/2 * omega_y 1/2 * q0 -1/2 * q3 1/2 * q2;
            1/2 * omega_y -1/2 * omega_z 0 1/2 * omega_x 1/2 * q3 1/2 * q0 -1/2 * q1;
            1/2 * omega_z 1/2 * omega_y -1/2 * omega_x 0 -1/2 * q2 1/2 * q1 1/2 * q0;
            0 0 0 0 0 (Iy - Iz)/Ix * omega_z (Iy- Iz)/Ix * omega_y;
            0 0 0 0 (Iz - Ix)/Iy * omega_z 0 (Iz - Ix)/Iy * omega_x;
            0 0 0 0  (Ix-Iy)/Iz * omega_y (Ix-Iy)/Iz * omega_x 0 ]
end

function make_H(filter::Kalman_Filter, i)
    #= H matrix

    :param i: index of dcm
    =#
    q0 = filter.state[1]
    q1 = filter.state[2]
    q2 = filter.state[3]
    q3 = filter.state[4]
    if i == 1
        return [2q0 2q1 -2q2 -2q3 0 0 0;
                2q3 2q2 2q1 2q0 0 0 0;
                -2q2 2q0 2q3 2q2 0 0 0]
    elseif i == 2
        return[-2q3 2q2 2q1 -2q0 0 0 0;
                2q0 -2q1 2q2 -2q3 0 0 0;
                2q1 2q0 2q3 2q2 0 0 0]
    elseif i == 3
        return [2q2 2q3 2q0 2q1 0 0 0;
                -2q1 -2q0 2q3 2q2 0 0 0;
                2q0 -2q1 -2q2 2q3 0 0 0]
    end
end

function predict(filter::Kalman_Filter)
    A = make_A(filter)
    phi = expm(A*STEP)
    gamma = inv(A) * (phi-1) * B
    filter.variance = phi * filter.variance *  phi' + gamma * Q * gamma'
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
    z_estimated = H * filter.state
    z = dcm - z_estimated
    x_hat = K * z
    filter.variance = P
    filter.state += x_hat
end

function plot(time, x, estimate)

    fig = figure()
    ax = fig[:add_subplot](111)
    ax[:plot](time, x[:,5], label=L"$\omega_x$")
    ax[:plot](time, x[:,6], label=L"$\omega_y$")
    ax[:plot](time, x[:,7], label=L"$\omega_z$")
    ax[:plot](time, estimate[:,5], label=L"estimated $\omega_x$")
    ax[:plot](time, estimate[:,6], label=L"estimated $\omega_y$")
    ax[:plot](time, estimate[:,7], label=L"estimated $\omega_z$")
    ax[:set_xlim]([0, last(time)])
    ax[:set_xlabel]("time [sec]")
    ax[:set_ylabel](L"$\omega$ [rad/s]")
    legend(loc = "best", fontsize=15)
    # PyPlot.plt[:show]()
    # PyPlot.plt[:savefig]("omega.pgf")


    fig = figure()
    ax = fig[:add_subplot](111)
    ax[:plot](time, x[:,1], label=L"$q_0$")
    ax[:plot](time, x[:,2], label=L"$q_1$")
    ax[:plot](time, x[:,3], label=L"$q_2$")
    ax[:plot](time, x[:,4], label=L"$q_3$")
    ax[:plot](time, estimate[:,1], label=L"estimated $q_0$")
    ax[:plot](time, estimate[:,2], label=L"estimated $q_1$")
    ax[:plot](time, estimate[:,3], label=L"estimated $q_2$")
    ax[:plot](time, estimate[:,4], label=L"estimated $q_3$")
    ax[:set_xlim]([0, last(time)])
    ax[:set_xlabel]("time [sec]")
    ax[:set_ylabel]("Quaternion")
    legend(loc = "best", fontsize=15)
    PyPlot.plt[:show]()
    # PyPlot.plt[:savefig]("quaternion.pgf")
end

function main()
    true_value = Array{Float64, 2}(STEPNUM+1,7)
    estimated_value = Array{Float64, 2}(STEPNUM+1,7)
    # initial condition
    true_value[1, 1:4] = Quaternion_ini
    true_value[1, 5:7] = omega_b

    estimated_value[1, :] = [rand_normal(0,0.01)  for x in 1:7]'
    estimated_value[1, 1:4] += Quaternion_ini
    estimated_value[1, 5:7] += omega_b
    kalman = Kalman_Filter(estimated_value[1,:],P_ini)
    time = zeros(STEPNUM+1)
    for i in 1:STEPNUM
        time[i+1] = i * STEP
        true_value[i+1, :] = true_value[i, :] +
                    runge_kutta(x -> differential_eq(x, 1),true_value[i,:], STEP)
        if  i % 100 == 99
            #observation and update
            index = rand(1:3)
            dcm = make_dcm(true_value,1)[index]
            update(kalman, dcm, index)
        else
            predict(kalman)
        end
        estimated_value[i+1, :] = kalman.state
    end
    plot(time, true_value,estimated_value)
end

main()
