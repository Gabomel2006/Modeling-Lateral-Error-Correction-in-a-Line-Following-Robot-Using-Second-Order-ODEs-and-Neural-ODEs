clc; clear; close all;

% ----------------------------------------------------------------
% ANALYTICAL PARTICULAR SOLUTION
% e(t) = e^(-t)(0.1*cos(sqrt(3)*t) + (0.1/sqrt(3))*sin(sqrt(3)*t))
% ----------------------------------------------------------------
t = linspace(0, 8, 1000);
alpha = -1;
beta  = sqrt(3);
C1    = 0.1;
C2    = 0.1 / sqrt(3);

e = exp(alpha*t) .* (C1*cos(beta*t) + C2*sin(beta*t));

% ----------------------------------------------------------------
% PLOT
% ----------------------------------------------------------------
figure(1);
hold on;

plot(t, e, 'b-', 'LineWidth', 2.5, ...
    'DisplayName', 'Particular solution $e(t)$');

yline(0, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2, ...
    'DisplayName', 'Target: $e(t) = 0$');

% ----------------------------------------------------------------
% FORMATTING
% ----------------------------------------------------------------
xlabel('Time $t$ (seconds)', 'FontSize', 13, 'Interpreter', 'latex');
ylabel('Lateral Error $e(t)$ (meters)', 'FontSize', 13, 'Interpreter', 'latex');
title('Particular Solution --- Underdamped Case ($k_d = 2$)', ...
    'FontSize', 14, 'Interpreter', 'latex');
legend('Location', 'best', 'FontSize', 11, 'Interpreter', 'latex');
grid on;
xlim([0 8]);
ylim([-0.08 0.12]);
hold off;

% ----------------------------------------------------------------
% SAVE
% ----------------------------------------------------------------
exportgraphics(gcf, 'particular_solution.png', 'Resolution', 300);
fprintf('Saved as particular_solution.png\n');