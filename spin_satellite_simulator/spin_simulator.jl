# Author:Shun Arahata
using PyPlot
import Elliptic.Jacobi
import Elliptic.Elliptic
const Ix = 1.9
const Iy = 1.6
const Iz = 2.0
rpm2radpersec(rpm) = rpm *2pi / 60
const ωs = rpm2radpersec(17)
const STEPNUM = 10000
const STEP = 0.01

q_initial = [1.0; 0.0; 0.0; 0.0]
ωb = [0.1; ωs + 0.1; 0.0]

function runge_kutta(f, t, x, step)
    #=　Fourth-order Runge-Kutta method.

    :param f: differential equation f(t,x)
     Note: input output must be the same dimension list
    :param t: variable
    :param x: variable
    :param step: step time
    :return: increment
    =#
    k1 = f(t, x)
    k2 = f(t + step/2, x + step / 2 * k1)
    k3 = f(t + step/2, x + step / 2 * k2)
    k4 = f(t + step/2, x + step * k3)
    sum = step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return sum
end

function differential_eq(t, x)
    q0 = x[1]
    q1 = x[2]
    q2 = x[3]
    q3 = x[4]
    ωx = x[5]
    ωy = x[6]
    ωz = x[7]
    dq0 = 1/2 * (-q1 * ωx + -q2 * ωy - q3 * ωz)
    dq1 = 1/2 * (q0 * ωx + -q3 * ωy + q2 * ωz)
    dq2 = 1/2 * (q3 * ωx + q0 * ωy - q1 * ωz)
    dq3 = 1/2 * (-q2 * ωx + q1 * ωy + q0 * ωz)
    norm = sqrt(dq0^2 + dq1^2 + dq2^2 + dq3^2)
    omega_x = (Iy - Iz)/Ix * ωy * ωz
    omega_y = (Iz - Ix)/Iy * ωz * ωx
    omega_z = (Ix - Iy)/Iz * ωx * ωy
    return [dq0/norm; dq1/norm; dq2/norm; dq3/norm; omega_x; omega_y; omega_z]
end

function make_theoretical_answer()
    #=
    :param ω: list of ωx, ωy, ωz
    =#
    ω_list = Array{Float64, 2}(STEPNUM+2,3)
    ω_list[1, :] = ωb'
    E = 0.5 * (Ix * ωb[1]^2 + Iy * ωb[2]^2 + Iz * ωb[3]^2)
    L2 = Ix^2 * ωb[1]^2 + Iy^2 * ωb[2]^2 + Iz^2 * ωb[3]^2
    lambda = sqrt((L2 - 2E * Iz)*(Iy - Ix)/(Ix * Iy * Iz))
    k2 = ((Ix - Iz)*(2E * Iy - L2))/((Iy - Ix)*(L2 - 2E * Iz))
    k = sqrt(k2)
    #find  period T
    period = 4 * Elliptic.K(k) / lambda
    # find t when omega_x = 0
    STEPDIV = 1000
    t_prime = 0
    abs_omega_x = 100
    for i in 0:STEPDIV
        t = i * period/STEPDIV
        omega_x = sqrt((2E*Iy - L2)/(Ix * (Iy - Ix))) * Jacobi.sn(lambda * t, k2)
        if abs(omega_x-ωb[1]) < abs_omega_x
            abs_omega_x = abs(omega_x-ωb[1])
            t_prime = t
        end
    end
    for i in 0:STEPNUM
        t = i * STEP
        omega_x = sqrt((2E*Iy - L2)/(Ix * (Iy - Ix))) * Jacobi.sn(lambda * (t + t_prime), k2)
        omega_y = sqrt((2E*Iz - L2)/(Iy * (Iz - Iy))) * Jacobi.dn(lambda * (t + t_prime), k2)
        omega_z = -sqrt((2E*Iy - L2)/(Iz * (Iy - Iz))) * Jacobi.cn(lambda * (t + t_prime), k2)
        ω_list[i+2, :] = [omega_x, omega_y, omega_z]
    end
    return ω_list
end

function plot(time, ω, q)

    fig = figure()
    ax = fig[:add_subplot](111)
    ax[:plot](time, ω[:,1], label=L"$\omega_x$")
    ax[:plot](time, ω[:,2], label=L"$\omega_y$")
    ax[:plot](time, ω[:,3], label=L"$\omega_z$")
    ax[:set_xlim]([0, last(time)])
    ax[:set_xlabel]("time [sec]")
    ax[:set_ylabel](L"$\omega$ [rad/s]")
    legend(loc = "best", fontsize=15)
    # PyPlot.plt[:show]()
    PyPlot.plt[:savefig]("omega.pgf")


    fig = figure()
    ax = fig[:add_subplot](111)
    ax[:plot](time, q[:,1], label=L"$q_0$")
    ax[:plot](time, q[:,2], label=L"$q_1$")
    ax[:plot](time, q[:,3], label=L"$q_2$")
    ax[:plot](time, q[:,4], label=L"$q_3$")
    ax[:set_xlim]([0, last(time)])
    ax[:set_xlabel]("time [sec]")
    ax[:set_ylabel]("Quaternion")
    legend(loc = "best", fontsize=15)
    PyPlot.plt[:savefig]("quaternion.pgf")

    fig = figure()
    theo = make_theoretical_answer()
    ax = fig[:add_subplot](311)
    ax[:plot](time, ω[:,1], label="Numerical")
    ax[:plot](time, theo[:,1], label="Theoretical")
    ax[:set_xlim]([0, last(time)])
    ax[:set_xlabel]("time [sec]")
    ax[:set_ylabel](L"$\omega_x$")
    legend(loc = 1)
    ax = fig[:add_subplot](312)
    ax[:plot](time, ω[:,2], label="Numerical")
    ax[:plot](time, theo[:,2], label="Theoretical")
    ax[:set_xlim]([0, last(time)])
    ax[:set_xlabel]("time [sec]")
    ax[:set_ylabel](L"$\omega_y$")
    legend(loc = 1)
    ax = fig[:add_subplot](313)
    ax[:plot](time, ω[:,3], label="Numerical")
    ax[:plot](time, theo[:,3], label="Theoretical")
    ax[:set_xlim]([0, last(time)])
    ax[:set_xlabel]("time [sec]")
    ax[:set_ylabel](L"$\omega_z$")
    legend(loc = 1)
    PyPlot.plt[:savefig]("theoretical.pgf")
    PyPlot.plt[:show]()
end

function main()
    x = Array{Float64, 2}(STEPNUM+2,7)
    x[1, 5:7] = ωb
    x[1, 1:4] = q_initial
    time = zeros(STEPNUM+2)
    for i in 0:STEPNUM
        x[i+2, :] = x[i+1, :] + runge_kutta(differential_eq, i * STEP, x[i+1,:], STEP)
        time[i+2] = (i+1) * STEP
    end
    # writecsv("quaternion.csv",q_list)
    plot(time, x[:,5:7], x[:,1:4])
end

@time main()
