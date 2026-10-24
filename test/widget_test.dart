import 'package:flutter_test/flutter_test.dart';
import 'package:sound_alert_app/main.dart';

void main() {
  testWidgets('Hearo app basic smoke test and splash navigation', (WidgetTester tester) async {
    // Build app widget tree
    await tester.pumpWidget(const HearoApp());

    // Verify splash screen branding is shown
    expect(find.text('Hearo'), findsOneWidget);
    expect(find.text('Every Sound Matters.'), findsOneWidget);

    // Let the splash transition finish and settle on the LoginScreen
    await tester.pump(const Duration(milliseconds: 2500));
    await tester.pumpAndSettle();

    // Verify Login Screen appears with Welcome Back and Hearo branding
    expect(find.text('Welcome Back'), findsOneWidget);
    expect(find.text('Hearo'), findsOneWidget);
  });
}
