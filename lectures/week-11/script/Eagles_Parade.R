## The Eagles Parade Problem: Temporal Validation vs. Rare Events
# Note: I asked Claude to write the code illustrating this problem. That's why it looks like this!

# When was the parade?
parade_date <- as.Date("2025-02-14")
parade_week <- week(parade_date)

cat("🦅 Eagles Super Bowl Parade:\n")
cat("  Date: February 14, 2025\n")
cat("  Week of year:", parade_week, "\n\n")

cat("Our temporal split:\n")
cat("  Training: weeks 1-9 (includes parade) ✓\n")
cat("  Test: weeks 10-13 (NO parade) ✗\n\n")

# Add parade feature to training data
train <- train %>%
  mutate(eagles_parade = ifelse(date == parade_date, 1, 0))

test <- test %>%
  mutate(eagles_parade = 0)  # No parade in test period

cat("Parade indicator distribution:\n")
cat("  Training set - parade day observations:", sum(train$eagles_parade), "\n")
cat("  Test set - parade day observations:", sum(test$eagles_parade), "\n\n")

# Fit model with parade feature
model1_parade <- lm(
  Trip_Count ~ as.factor(hour) + dotw_simple + Temperature + 
    Precipitation + eagles_parade,
  data = train
)

# Show the coefficient
parade_coef <- summary(model1_parade)$coefficients["eagles_parade", ]
cat("Eagles Parade Coefficient (learned from training data):\n")
cat("  Estimate:", round(parade_coef[1], 3), "trips per station-hour\n")
cat("  p-value:", format.pval(parade_coef[4], digits = 3), "\n\n")

# Here's the problem!
cat("⚠️  THE PROBLEM:\n")
cat("We learned a parade effect from training data, but we CAN'T measure\n")
cat("whether it generalizes because there's no parade in our test period!\n\n")

# Test set performance won't change
test$pred1_parade <- predict(model1_parade, newdata = test)

mae_original <- mean(abs(test$Trip_Count - test$pred1), na.rm = TRUE)
mae_with_parade <- mean(abs(test$Trip_Count - test$pred1_parade), na.rm = TRUE)

cat("Test set MAE:\n")
cat("  Without parade feature:", round(mae_original, 3), "\n")
cat("  With parade feature:", round(mae_with_parade, 3), "\n")
cat("  Difference:", round(abs(mae_original - mae_with_parade), 4), 
    "(essentially zero!)\n\n")

cat("Why no difference? Because test set has eagles_parade = 0 for all observations!\n")
cat("The parade coefficient is learned but never 'activated' in test predictions.\n")