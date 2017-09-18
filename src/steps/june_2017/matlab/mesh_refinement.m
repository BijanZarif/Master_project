%% x component of velocity in x_0

fig = figure

hold on
grid on
plot(time_4, v_x0_4, '--', 'linewidth',2)
plot(time_8, v_x0_8, 'linewidth',2)
plot(time_16, v_x0_16,'--', 'linewidth',2)
ylabel('v_x_0')
xlabel('time');
legend('N=2^2','N=2^3','N=2^4')
title('x component of velocity in x_0')