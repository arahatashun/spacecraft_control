# Author:Shun Arahata
using PyPlot
const Ix = 1.9
const Iy = 1.6
const Iz = 2.0
rpm2radpersec(rpm) = rpm *2pi / 60
const ωs = rpm2radpersec(17)
const STEPNUM = 10000
const STEP = 0.01

q_initial = [1, 0, 0, 0]
ωb = [0.1,ωs + 0.1, 0.0]

function runge_kutta(f, t, x, step)
    #=　Fourth-order Runge-Kutta method.

    :param f: differential equation f(t,x)
    :param t: variable
    :param x: variable
    :param step: step time
    :return: increment
    =#
    k1 = f(t, x)
    k2 = f(t + step/2, x + step / 2 * k1)
    k3 = f(t + step/2, x + step / 2 * k2)
    k4 = f(t + step/2, x + step * k3)
    return step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
end

function ω_differential(t, ω)
    #= differential equation of ω.

    :param t: time(not related)
    :param ω: list of ωx, ωy, ωz
    :return: differential of ω list
    =#
    ωx = (Iy - Iz)/Ix * ω[2] * ω[3]
    ωy = (Iz - Ix)/Iy * ω[3] * ω[1]
    ωz = (Ix - Iy)/Iz * ω[1] * ω[2]
    return [ωx, ωy, ωz]
end

function quaternion_differntial(t, qandω)
    #= differential equation of quaternion.

    :param t: time(not related)
    :param qandω: list of quaternion + ω
    :return: differential of quaternion
    =#
    q0 = qandω[1]
    q1 = qandω[2]
    q2 = qandω[3]
    q3 = qandω[4]
    ωx = qandω[5]
    ωy = qandω[6]
    ωz = qandω[7]
    dq0 = 1/2(-q1 * ωx + -q2 * ωy - q3 * ωz)
    dq1 = 1/2(q0 * ωx + -q3 * ωy + q2 * ωz)
    dq2 = 1/2(q3 * ωx + q0 * ωy - q1 * ωz)
    dq3 = 1/2(-q2 * ωx + q1 * ωy + q0 * ωz)
    return [dq0, dq1, dq2, dq3, dq4]
end

function main()
    ω_list = Array{Float64, 2}(STEPNUM+2,3)
    ω_list[1, :] = ωb'
    ω_new = ωb
    time = [0.0]
    for i in 0:STEPNUM
        ω_new += runge_kutta(ω_differential, i * STEP, ω_new, STEP)
        push!(time, (i+1) * STEP)
        ω_list[i+2, :] = ω_new'
    end
    fig = figure()
    ax = fig[:add_subplot](111)
    ax[:plot](time, ω_list[:,1], label=L"$\omega_x$")
    ax[:plot](time, ω_list[:,2], label=L"$\omega_y$")
    ax[:plot](time, ω_list[:,3], label=L"$\omega_z$")
    legend(loc = "best", fontsize=15)
    PyPlot.plt[:show]()
end

main()
