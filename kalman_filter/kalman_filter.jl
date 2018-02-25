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

Quaternion_ini = [1.0; 0.0; 0.0; 0.0]
omega_b = [0.1; omega_s + 0.1; 0.0]
B = [0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0; 1/Ix, 0, 0;0, 1/Iy, 0; 0, 0, 1/Iz]
q_std = 0.01
Q = [q_std^2 ,0 , 0;0, q_std^2, 0; 0, 0, q_std^2]
r_std = 0.01
R = [r_std^2, 0, 0;0, r_std^2, 0;0, 0, r_std^2]
P_ini = [0.01, 0, 0, 0, 0, 0, 0;0, 0.01, 0, 0, 0, 0, 0;
    0, 0, 0.01, 0, 0, 0, 0;0, 0, 0, 0.01, 0, 0, 0;
    0, 0, 0, 0, 0, 0.01, 0;0, 0, 0, 0, 0, 0, 0.01]

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
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    x = [q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2 + noise * rand_normal(0,r_std);
        2 * (q1 * q2 - q0 * q3) + noise * rand_normal(0,r_std);
        2 * (q1 * q3 + q0 * q2) + noise * rand_normal(0,r_std)]

    y = [2 * (q1 * q2 + q0 * q3) + noise * rand_normal(0,r_std);
        q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2 + noise * rand_normal(0,r_std);
        2 * (q2 * q3 - q0 * q1) + noise * rand_normal(0,r_std)]

    z = [2 * (q1 * q3 - q0 * q2) + noise * rand_normal(0,r_std);
        2 * (q2 * q3 + q0 * q1) + noise * rand_normal(0,r_std);
        q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2] + noise * rand_normal(0,r_std))

    return x, y, z
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


type Kalman_Filter
    state::Array{Float64, 2}(1,7)
    variance::Array{Float54,2}(7,7)
end

function A(filter::Kalman_Filter)
    x = filter.state
    q0 = x[1]
    q1 = x[2]
    q2 = x[3]
    q3 = x[4]
    omega_x = x[5]
    omega_y = x[6]
    omega_z = x[7]
    return  [0, -1/2*omega_x, -1/2*omega_y, -1/2*omega_z, -1/2 * q1, -1/2q2, 1/2q3;
            1/2*omega_x, 0, 1/2*omega_z, -1/2*omega_y, 1/2 * q0. -1/2q3, 1/2q2;
            0, 0, 0, 0, 0, (Iy-Iz)/Ix*oemga_z, (Iy-Iz)/Ix*omega_y];
end

function predict(filter::Kalman_Filter)
    A = A(filter)
    phi = expm(A*STEP)
    gamma = inv(A) * (phi-1) * B
    filter.variance = gamma * filter.variance *  gamma' + gamma * Q * gamma'
    filter.state += runge_kutta(x -> differential_eq(x, 0), filter.state, STEP)
end

function update(filter::Kalman_Filter,index::Int)
    #=observation and update step

    :param index: index of dcm vector
    =#

end

function main()
    true_value = Array{Float64, 2}(STEPNUM+1,7)
    estimated_value = Array{Kalman_Filter, 2}(STEPNUM+1,7)
    # initial condition
    true_value[1, 1:4] = q_initial
    true_value[1, 5:7] = omega_b

    estimated_value[1, :] = [rand_normal(0,0.01)  for x in 1:7]'
    estimated_value[1, 1:4] += Quaternion_ini
    estimated_value[1, 5:7] += omega_b
    kalman = Kalman_Filter(estimated_value, )
    time = zeros(STEPNUM+1)
    for i in 1:STEPNUM
        time[i+1] = i * STEP
        true_value[i+1, :] = true_value[i, :] +
                    runge_kutta(x -> differential_eq(x, 1),true_value[i,:], STEP)
        if STEPNUM % 100 == 0
            #observation and update

        else
            predict(kalman)
        end
    end
