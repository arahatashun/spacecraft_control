# Author:Shun Arahata
# kalman filter
using PyPlot
const Ix = 1.9
const Iy = 1.6
const Iz = 2.0
rpm2radpersec(rpm) = rpm *2pi / 60
const ωs = rpm2radpersec(17)
const STEPNUM = 10000
const STEP = 0.01

true_value = Array{Float64, 2}(STEPNUM+1,7)
estimated_value = Array{Float64, 2}(STEPNUM+1,7)
q_initial = [1.0; 0.0; 0.0; 0.0]
ωb = [0.1; ωs + 0.1; 0.0]
B = [0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 1/Ix, 0, 0;0, 1/Iy, 0; 0, 0, 1/Iz]
P_ini = [0.01, 0, 0, 0, 0, 0, 0;
    0, 0.01, 0, 0, 0, 0, 0;
    0, 0, 0.01, 0, 0, 0, 0;
    0, 0, 0, 0.01, 0, 0, 0;
    0, 0, 0, 0, 0, 0.01, 0;
    0, 0, 0, 0, 0, 0, 0.01]

A(omega_x, omega_y, omega_z, q0, q1, q2, q3)
    = [0, -1/2*omega_x, -1/2*omega_y, -1/2*omega_z, -1/2 * q1, -1/2q2, 1/2q3;
        1/2*omega_x, 0, 1/2*omega_z, -1/2*omega_y, 1/2 * q0. -1/2q3, 1/2q2;
        0, 0, 0, 0, 0, (Iy-Iz)/Ix*oemga_z, (Iy-Iz)/Ix*omega_y;
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

function make_dcm(q,noise)
    #= make dcm vector and (add noise)

    :param x:q
    :param noise: adding noise or not. 0 or 1
    :return :2 DCM vector
    =#
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    x = [q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2 + noise * rand_normal(0,0.01);
        2 * (q1 * q2 - q0 * q3) + noise * rand_normal(0,0.01);
        2 * (q1 * q3 + q0 * q2) + noise * rand_normal(0,0.01)]

    y = [2 * (q1 * q2 + q0 * q3) + noise * rand_normal(0,0.01);
        q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2 + noise * rand_normal(0,0.01);
        2 * (q2 * q3 - q0 * q1) + noise * rand_normal(0,0.01)]

    z = [2 * (q1 * q3 - q0 * q2) + noise * rand_normal(0,0.01);
        2 * (q2 * q3 + q0 * q1) + noise * rand_normal(0,0.01);
        q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2] + noise * rand_normal(0,0.01))

    return x, y, z
end

function differential_eq_with_noise(t, x)
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
    omega_x = (Iy - Iz)/Ix * ωy * ωz + rand_normal(0,0.01)/Ix
    omega_y = (Iz - Ix)/Iy * ωz * ωx + rand_normal(0,0.01)/Iy
    omega_z = (Ix - Iy)/Iz * ωx * ωy + rand_normal(0,0.01)/Iz
    return [dq0/norm; dq1/norm; dq2/norm; dq3/norm; omega_x; omega_y; omega_z]
end

function differential_eq_no_noise(t, x)
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


function main()
    # initial condition
    true_value[1, 1:4] = q_initial
    true_value[1, 5:7] = ωb
    estimated_value[1, :] = [rand_normal(0,0.01)  for x in 1:7]'
    estimated_value[1, 1:4] += q_initial
    estimated_value[1, 5:7] += ωb
    time = zeros(STEPNUM+1)
    for i in 1:STEPNUM
        time[i+1] = i * STEP
        true_value[i+1, :] = true_value[i, :] +
                    runge_kutta(differential_eq_with_noise, i * STEP, true_value[i,:], STEP)
        if STEPNUM % 100 == 0

        else
            estimated_value[i+1, :] = estimated_value[i, :] +
                runge_kutta(differential_eq_no_noise, i * STEP, true_value[i,:], STEP)
        end
    end
