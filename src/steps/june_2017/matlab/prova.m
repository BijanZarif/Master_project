fig = figure

hold on
grid on
plot(time, flux_0_0001, '--', 'linewidth',2)
plot(time, flux_0_001, 'linewidth',2)
ylabel('flux [dimension]')
xlabel('time');
legend('0.0001','0.001')
title('Flux')
