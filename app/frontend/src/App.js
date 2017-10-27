import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { withStyles } from 'material-ui/styles';
import Grid from 'material-ui/Grid';
import Paper from 'material-ui/Paper';
import AppBar from 'material-ui/AppBar';
import Toolbar from 'material-ui/Toolbar';
import Typography from 'material-ui/Typography';

const styles = theme => ({
    root: {
        flexGrow: 1,
        overflow: "hidden",
        textAlign: "center",
        //marginTop: 64,

    },
    flex: {
        flex: 1,
    },
    paper: {
        textAlign: 'center',
        color: theme.palette.text.secondary,
    },
});

/**
 * The app which consists of ...
 */
class App extends Component
{
    render() {
        const { classes } = this.props;

        return (
            <div className={classes.root}>
                <AppBar className={classes.appbar} position="static" color="default">
                    <Toolbar>
                        <Typography className={classes.flex} type="title" color="inherit">
                            Deep Lyrics
                        </Typography>
                    </Toolbar>
                </AppBar>
                <Grid container className={classes.container}>
                    <Grid item xs={12} sm={6}>
                        <Paper className={classes.paper}>
                        </Paper>
                    </Grid>
                </Grid>
            </div>
        );
    }
}

App.propTypes = {
    classes: PropTypes.object.isRequired,
};

export default withStyles(styles)(App);

