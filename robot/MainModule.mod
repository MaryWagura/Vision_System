MODULE MainModule
    !=====================================================================
    ! INA700 vision pick-and-place - ABB main module
    !
    ! Frames:  every move uses tool1234 \WObj:=wobj1212.
    ! Python must send pick X/Y in wobj1212:
    !   MARKER_ROBOT_X_MM = aruco.trans.x, MARKER_ROBOT_Y_MM = aruco.trans.y
    !
    ! Before running after ANY tool or work object change:
    !   - re-teach aruco (cup touching the marker centre)
    !   - re-teach p20 / p30 / p40 (place positions) in wobj1212
    !   - first run in Manual reduced speed, stepping one instruction
    !=====================================================================

    TASK PERS tooldata tool1234:=[TRUE,[[-96.07,0.34481,181.882],[1,0,0,0]],[0.5,[50,0,50],[1,0,0,0],0,0,0]];
    TASK PERS wobjdata wobj1212:=[FALSE,TRUE,"",[[448.359,-106.848,-10.0174],[0.999994,-0.00313365,0.00128173,-0.00037779]],[[0,0,0],[1,0,0,0]]];

    ! Home: high above the paper, near the marker
    CONST robtarget p10:=[[167.26,188.34,180.37],[0.0362958,0.0304525,-0.998773,-0.0144165],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];

    ! Place positions on the designated-positions board.
    ! WARNING: these values were taught in the old wobj1234, whose origin
    ! is ~30 mm away from wobj1212. Re-teach them in wobj1212.
    CONST robtarget p20:=[[273.73,-66.71,14.75],[0.0110135,-0.0258492,-0.999605,0.000494662],[-1,-1,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget p30:=[[205.88,-136.26,15.07],[0.0110339,-0.0258024,-0.999606,0.000509208],[-1,-1,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget p40:=[[261.03,-195.60,14.50],[0.0110429,-0.0257536,-0.999607,0.000508994],[-1,-1,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];

    ! Taught with the cup on the ArUco marker centre (wobj1212).
    ! Its X/Y must match MARKER_ROBOT_X_MM / MARKER_ROBOT_Y_MM in Python.
    CONST robtarget aruco:=[[151.16,202.04,2.05],[0.0363271,0.0305793,-0.998768,-0.0143814],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];

    ! --- Heights (mm, wobj1212) ---
    CONST num PAPER_Z := 2.05;      ! Z with the cup on the paper (= aruco.trans.z)
    CONST num SHAPE_THICK := 10;    ! all shapes are 1 cm thick
    CONST num SQUEEZE := 0;         ! mm to press into the cup; raise by 0.5 if no seal
    CONST num MIN_Z := 4;           ! crash guard: never closer to the paper
    CONST num APPROACH_DZ := 80;    ! hover height above pick/place before descending
    CONST num PLACE_DZ := 5;        ! release height above p20-p40; 0 once re-taught

    ! --- Allowed pick area, relative to the marker centre (mm) ---
    ! Adjust to your paper. Targets outside are skipped, never moved to.
    CONST num REACH_X := 200;
    CONST num REACH_Y := 140;

    ! Set FALSE to skip the hover-over-marker calibration check
    CONST bool SHOW_ARUCO := TRUE;

    PROC PickAndPlace(robtarget pickTarget, robtarget placeTarget)
        MoveJ p10, v500, fine, tool1234\WObj:=wobj1212;

        IF SHOW_ARUCO THEN
            ! Visual check: cup should hover exactly over the marker centre
            MoveJ Offs(aruco, 0, 0, APPROACH_DZ), v200, fine, tool1234\WObj:=wobj1212;
        ENDIF

        TPWrite "Moving to pick object";
        MoveJ Offs(pickTarget, 0, 0, APPROACH_DZ), v200, fine, tool1234\WObj:=wobj1212;
        MoveL pickTarget, v50, fine, tool1234\WObj:=wobj1212;

        TPWrite "Vacuum ON";
        Set doValve1;
        WaitTime 0.5;
        MoveL Offs(pickTarget, 0, 0, APPROACH_DZ), v100, fine, tool1234\WObj:=wobj1212;

        MoveJ p10, v500, fine, tool1234\WObj:=wobj1212;

        TPWrite "Moving to placing position";
        MoveJ Offs(placeTarget, 0, 0, APPROACH_DZ), v200, fine, tool1234\WObj:=wobj1212;
        MoveL Offs(placeTarget, 0, 0, PLACE_DZ), v50, fine, tool1234\WObj:=wobj1212;

        TPWrite "Vacuum OFF";
        Reset doValve1;
        WaitTime 1;
        MoveL Offs(placeTarget, 0, 0, APPROACH_DZ), v100, fine, tool1234\WObj:=wobj1212;

        MoveJ p10, v500, fine, tool1234\WObj:=wobj1212;
    ENDPROC

    PROC main()
        VAR string visionData;
        VAR num comma1;
        VAR num comma2;
        VAR num comma3;
        VAR string str_shape;
        VAR string str_x;
        VAR string str_y;
        VAR string str_angle;
        VAR num val_x;
        VAR num val_y;
        VAR num val_angle;
        VAR bool ok_x;
        VAR bool ok_y;
        VAR bool ok_angle;
        VAR num pickZ;
        VAR bool known_shape;
        VAR robtarget dynamicPickTarget;
        VAR robtarget staticPlaceTarget;

        TPWrite "Vision Pick and Place Started";

        RobotAsClientConnect;

        WHILE TRUE DO
            TPWrite "Requesting targets from camera...";
            RobotClienSendMessage("REQUEST_COORDS");

            visionData := RobotClientReciveMessage();

            IF visionData = "" OR visionData = "NO_TARGET,0,0,0" THEN
                TPWrite "Camera sees no shapes. Waiting 2 seconds...";
                WaitTime 2;
            ELSE
                comma1 := StrFind(visionData, 1, ",");

                IF comma1 > 0 AND comma1 < StrLen(visionData) THEN
                    comma2 := StrFind(visionData, comma1 + 1, ",");
                ELSE
                    comma2 := 0;
                ENDIF

                IF comma2 > 0 AND comma2 < StrLen(visionData) THEN
                    comma3 := StrFind(visionData, comma2 + 1, ",");
                ELSE
                    comma3 := 0;
                ENDIF

                IF comma1 > 0 AND comma2 > 0 AND comma3 > 0 THEN
                    str_shape := StrPart(visionData, 1, comma1 - 1);
                    str_x := StrPart(visionData, comma1 + 1, comma2 - comma1 - 1);
                    str_y := StrPart(visionData, comma2 + 1, comma3 - comma2 - 1);
                    str_angle := StrPart(visionData, comma3 + 1, StrLen(visionData) - comma3);

                    ok_x := StrToVal(str_x, val_x);
                    ok_y := StrToVal(str_y, val_y);
                    ok_angle := StrToVal(str_angle, val_angle);
                    IF ok_angle = FALSE THEN
                        val_angle := 0;
                    ENDIF

                    TPWrite "Detected: " + str_shape;

                    ! Python sends lowercase names: circle / star / square
                    known_shape := TRUE;
                    IF str_shape = "circle" THEN
                        staticPlaceTarget := p20;
                    ELSEIF str_shape = "star" THEN
                        staticPlaceTarget := p30;
                    ELSEIF str_shape = "square" THEN
                        staticPlaceTarget := p40;
                    ELSE
                        known_shape := FALSE;
                    ENDIF

                    ! Pick height: top face of a 1 cm shape
                    pickZ := PAPER_Z + SHAPE_THICK - SQUEEZE;
                    IF pickZ < MIN_Z THEN
                        pickZ := MIN_Z;
                    ENDIF

                    ! In RAPID, NOT has the same low priority as OR, so
                    ! "NOT a OR NOT b" does not compile - compare to FALSE instead.
                    IF ok_x = FALSE OR ok_y = FALSE THEN
                        TPWrite "Bad coordinates received: " + visionData;
                        WaitTime 2;
                    ELSEIF known_shape = FALSE THEN
                        TPWrite "Unknown shape - skipped: " + str_shape;
                        WaitTime 2;
                    ELSEIF Abs(val_x - aruco.trans.x) > REACH_X OR Abs(val_y - aruco.trans.y) > REACH_Y THEN
                        TPWrite "Target outside pick area - skipped";
                        WaitTime 2;
                    ELSE
                        ! Base the pick on the aruco target: taught on the paper,
                        ! tool pointing down, arm configuration for the paper area
                        dynamicPickTarget := aruco;
                        dynamicPickTarget.trans := [val_x, val_y, pickZ];
                        dynamicPickTarget := RelTool(dynamicPickTarget, 0, 0, 0 \Rz:=val_angle);

                        TPWrite "Press yellow button to pick " + str_shape;
                        WaitDI diOkButton, 1;

                        PickAndPlace dynamicPickTarget, staticPlaceTarget;
                        TPWrite "Pick and place complete";
                    ENDIF
                ELSE
                    TPWrite "Bad data received: " + visionData;
                    WaitTime 2;
                ENDIF
            ENDIF
        ENDWHILE
    ENDPROC
ENDMODULE
